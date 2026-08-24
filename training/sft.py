import os
import sys
import json
import math
import time
import random
from pathlib import Path

import torch as th
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
CONFIG_PATH = REPO_ROOT / "config" / os.getenv("SFT_CONFIG", "sft.json")
CONFIG_DIR = CONFIG_PATH.parent

sys.path.insert(0, str(REPO_ROOT / "llm"))
from model import Model
from tokenizer import ByteTokenizer


def format_prompt(instruction: str, inp: str = "") -> str:
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


class InstructDataset(Dataset):
    """Flat packed-token dataset: all samples concatenated then chunked
    into fixed-length windows (no per-sample padding → efficient)."""

    def __init__(self, path, tok: ByteTokenizer, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        all_tokens = []
        n_samples = 0
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    text = format_prompt(r.get("instruction", ""), r.get("input", "") or "") + r.get("output", "")
                    all_tokens.extend(tok.tokenize(text))
                    n_samples += 1
        self.data = th.tensor(all_tokens, dtype=th.long)
        self.n = max(1, (len(self.data) - 1) // seq_len)
        print(f"  dataset {Path(path).name}: {n_samples} samples / {len(all_tokens):,} tokens -> {self.n} chunks")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.seq_len
        chunk = self.data[s: s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = th.cat([chunk, th.full((self.seq_len + 1 - len(chunk),), self.tok.pad_id)])
        return chunk


def collate(batch):
    return th.stack(batch)


def evaluate(mdl, loader, device, vocab_size, label_smoothing, steps=40):
    mdl.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = 0
    with th.no_grad():
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            with autocast("cuda", enabled=device == "cuda"):
                logits = mdl(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
                ignore_index=0,
                label_smoothing=label_smoothing,
            )
            total_loss += loss.item()
            pred = logits.argmax(-1)
            mask = y != 0
            total_correct += (pred[mask] == y[mask]).float().sum().item()
            total_tokens += mask.sum().item()
            n += 1
    mdl.train()
    return total_loss / max(n, 1), total_correct / max(total_tokens, 1)


def main():
    cfg = json.load(open(CONFIG_PATH))
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]

    random.seed(42)
    th.manual_seed(42)

    tok = ByteTokenizer()
    vocab_size = tok.vocab_size

    device = "cuda" if th.cuda.is_available() else "cpu"
    n_gpus = th.cuda.device_count()
    print(f"device={device} gpus={n_gpus}")

    mdl = Model(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        transformer_layers=mcfg["transformer_layers"],
        max_len=mcfg["max_len"],
    )
    mdl.lm_head.weight = mdl.embed.weight  # tie (matches finetune / generate)

    pretrained = (CONFIG_DIR / cfg["pretrained_weights"]).resolve()
    if pretrained.exists():
        ckpt = th.load(pretrained, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mdl.load_state_dict(sd)
        print(f"Loaded pretrained ✓ ({pretrained})")
    else:
        print("WARNING: pretrained weights not found, training from scratch")

    if n_gpus > 1:
        mdl = th.nn.DataParallel(mdl)
    mdl = mdl.to(device)

    def _core(m):
        return m.module if isinstance(m, th.nn.DataParallel) else m

    seq_len = mcfg["max_len"]
    train_ds = InstructDataset(REPO_ROOT / dcfg["train_file"], tok, seq_len)
    eval_ds = InstructDataset(REPO_ROOT / dcfg["eval_file"], tok, seq_len)
    if len(train_ds) == 0:
        raise SystemExit("No training data; run scripts/build_sft_data.py first.")

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate, drop_last=False
    )

    max_steps = tcfg["max_steps"]
    warmup = tcfg.get("warmup_steps", 0)
    base_lr = tcfg["lr"]
    lr_min = tcfg.get("lr_min", base_lr * 0.1)
    grad_accum = tcfg.get("grad_accum", 1)
    label_smoothing = tcfg.get("label_smoothing", 0.0)

    opt = th.optim.AdamW(_core(mdl).parameters(), lr=base_lr, weight_decay=0.05)
    scaler = GradScaler("cuda", enabled=device == "cuda")

    save_dir = (CONFIG_DIR / cfg["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "instruct_best.pt"
    log_path = save_dir / "instruct_log.jsonl"

    start = time.time()
    max_seconds = tcfg.get("max_hours", 6) * 3600.0

    step = 0
    best_eval = float("inf")
    patience = tcfg.get("overfit_patience", 4)
    bad_epochs = 0
    time_up = False
    opt.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps or time.time() - start > max_seconds:
                time_up = True
                break
            step += 1
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            with autocast("cuda", enabled=device == "cuda"):
                logits = mdl(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                    ignore_index=0,
                    label_smoothing=label_smoothing,
                )
                loss = loss / grad_accum
            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.unscale_(opt)
                th.nn.utils.clip_grad_norm_(_core(mdl).parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if step < warmup:
                    lr = base_lr * (step / max(warmup, 1))
                else:
                    prog = (step - warmup) / max(1, max_steps - warmup)
                    lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * min(1.0, prog)))
                for g in opt.param_groups:
                    g["lr"] = float(lr)

            if step % tcfg.get("log_every", 50) == 0:
                print(f"step={step} loss={loss.item()*grad_accum:.4f} lr={opt.param_groups[0]['lr']:.2e} t={(time.time()-start)/60:.1f}m")

            if step % tcfg.get("eval_every", 500) == 0:
                el, acc = evaluate(mdl, eval_loader, device, vocab_size, label_smoothing)
                print(f"[eval] step={step} eval_loss={el:.4f} token_acc={acc:.3f}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "eval_loss": el, "token_acc": acc}) + "\n")
                if el < best_eval:
                    best_eval = el
                    bad_epochs = 0
                    th.save({"model": _core(mdl).state_dict(), "step": step, "loss": el}, best_path)
                    print(f"  new best -> {best_path}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"  early-stop: eval increased {bad_epochs} times. best_eval={best_eval:.4f}")
                        time_up = True
                        break

        if time_up:
            break

    th.save({"model": _core(mdl).state_dict(), "step": step, "loss": best_eval}, save_dir / "instruct_last.pt")
    print(f"DONE step={step} best_eval={best_eval:.4f} | best={best_path}")


if __name__ == "__main__":
    main()
