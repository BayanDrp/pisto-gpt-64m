# ============================================================
# Pretraining — loads all settings from config/train.json
# ============================================================
import sys, os, math, time, json, random
from pathlib import Path
import multiprocessing
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

# Force line-buffered output so we can see progress/errors
sys.stdout.reconfigure(line_buffering=True)

import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Load config ──────────────────────────────────────────────
_HERE = Path(__file__).parent
CONFIG_PATH = _HERE.parent / "config" / os.getenv("PRETRAIN_CONFIG", "train.json")
CONFIG_DIR = CONFIG_PATH.parent

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

model_cfg   = cfg["model"]
train_cfg   = cfg["training"]
dataset_cfg = cfg["dataset"]
SAVE_DIR    = (CONFIG_DIR / cfg["save_dir"]).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN    = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ── Model path ───────────────────────────────────────────────
sys.path.insert(0, str(_HERE.parent / "llm"))
from model import Model as LLMModel
from tokenizer import ByteTokenizer

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device : {device}")
if th.cuda.is_available():
    print(f"GPU    : {th.cuda.get_device_name(0)}")
print(f"Save   : {SAVE_DIR}")
print(f"Config : {CONFIG_PATH}")

# ── Model ────────────────────────────────────────────────────
mdl = LLMModel(
    d_model=model_cfg["d_model"],
    nhead=model_cfg["nhead"],
    dim_feedforward=model_cfg["dim_feedforward"],
    dropout=model_cfg["dropout"],
    transformer_layers=model_cfg["transformer_layers"],
    max_len=model_cfg["max_len"],
).to(device)

mdl.lm_head.weight = mdl.embed.weight

if th.cuda.is_available() and th.cuda.device_count() > 1:
    print(f"Using {th.cuda.device_count()} GPUs via DataParallel")
    mdl = th.nn.DataParallel(mdl)
_core = lambda m: m.module if isinstance(m, th.nn.DataParallel) else m

total = sum(p.numel() for p in mdl.parameters())
print(f"Parameters: {total/1e6:.1f}M")

# ── Dataset ──────────────────────────────────────────────────
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=512):
        self.seq_len = seq_len
        print(f"Tokenizing {len(texts):,} docs...")
        all_tokens = []
        for text in texts:
            all_tokens.extend(tokenizer.tokenize(text))
        self.data = th.tensor(all_tokens, dtype=th.long)
        self.n = (len(self.data) - 1) // seq_len
        if self.n == 0:
            raise ValueError(f"Not enough data: only {len(self.data)} tokens, need {seq_len + 1}")
        print(f"Tokens: {len(self.data):,} → {self.n:,} chunks")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = th.zeros(self.seq_len + 1 - len(chunk), dtype=th.long)
            chunk = th.cat([chunk, pad])
        return chunk

tokenizer = ByteTokenizer()

print(f"Loading {dataset_cfg['name']}...")
load_kwargs = {
    "path": dataset_cfg["name"],
    "split": dataset_cfg["split"],
    "streaming": True,
}
if HF_TOKEN:
    load_kwargs["token"] = HF_TOKEN
try:
    ds = load_dataset(**load_kwargs)
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    raise

texts = []
for i, s in enumerate(ds):
    if i >= dataset_cfg["max_docs"]:
        break
    text = s.get("text", "")
    if len(text) > dataset_cfg["min_len"]:
        texts.append(text)
    if i % 20_000 == 0:
        print(f"  {i:,} docs...")

random.shuffle(texts)
split = int(len(texts) * dataset_cfg["train_split"])

seq_len = model_cfg["max_len"]
batch_size = train_cfg["batch_size"]

train_ds = TextDataset(texts[:split], tokenizer, seq_len=seq_len)
eval_ds  = TextDataset(texts[split:], tokenizer, seq_len=seq_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=th.cuda.is_available(), drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=th.cuda.is_available(), drop_last=True)

print("Dataset ready ✓")

# ── Training settings ────────────────────────────────────────
MAX_HOURS    = train_cfg["max_hours"]
GRAD_ACCUM   = train_cfg["grad_accum"]
MAX_STEPS    = train_cfg["max_steps"]
WARMUP_STEPS = train_cfg["warmup_steps"]
LOG_EVERY    = train_cfg["log_every"]
EVAL_EVERY   = train_cfg["eval_every"]
LR           = train_cfg["lr"]
LR_MIN       = train_cfg["lr_min"]

# ── Overfitting detection ────────────────────────────────────
OVERFIT_PATIENCE = 5       # stop after N consecutive eval-loss increases
eval_history     = []
done             = False   # shared stop flag

# ── Optimizer ────────────────────────────────────────────────
decay, no_decay = [], []
for name, p in mdl.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() < 2 or any(x in name for x in ["ln", "bias", "embed"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = th.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=LR, betas=(0.9, 0.95), fused=th.cuda.is_available())

scaler = th.amp.GradScaler("cuda", enabled=th.cuda.is_available())

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return LR_MIN + (LR - LR_MIN) * cosine

@th.no_grad()
def evaluate(n=30):
    mdl.eval()
    total = 0
    n_batches = 0
    try:
        for i, batch in enumerate(eval_loader):
            if i >= n:
                break
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            with th.amp.autocast("cuda", enabled=th.cuda.is_available()):
                loss = F.cross_entropy(
                    mdl(x).reshape(-1, _core(mdl).vocab_size),
                    y.reshape(-1),
                    ignore_index=0,
                )
            total += loss.item()
            n_batches += 1
    finally:
        mdl.train()
    return total / max(n_batches, 1)

def save_ckpt(step, loss):
    path = SAVE_DIR / "pretrain_best.pt"
    th.save({
        "step": step,
        "loss": loss,
        "model": _core(mdl).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ pretrain_best.pt saved (step={step:,})")

# ── Resume ───────────────────────────────────────────────────
step      = 0
best_eval = float("inf")
loss_val  = 0.0

resume_path = SAVE_DIR / "pretrain_best.pt"
if resume_path.exists():
    print(f"Resuming from {resume_path}...")
    ckpt = th.load(resume_path, map_location=device, weights_only=False)
    _core(mdl).load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    best_eval = ckpt.get("loss", float("inf"))
    print(f"Resumed: step={step:,} | best_eval={best_eval:.4f} | tokens={step*batch_size*seq_len*GRAD_ACCUM/1e6:.0f}M")
else:
    print("Starting fresh")

# ── Training loop ────────────────────────────────────────────
log_path = SAVE_DIR / "log.jsonl"
t0       = time.time()
mdl.train()

print(f"\n{'='*50}")
print(f"Pretraining | max {MAX_HOURS}h | overfit_patience={OVERFIT_PATIENCE}")
print(f"{'='*50}\n")

loader_iter = iter(train_loader)

while step < MAX_STEPS and not done:
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_val = 0.0

    for _ in range(GRAD_ACCUM):
        try:
            micro = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            micro = next(loader_iter)

        x = micro[:, :-1].to(device, non_blocking=True)
        y = micro[:, 1:].to(device, non_blocking=True)
        with th.amp.autocast("cuda", enabled=th.cuda.is_available()):
            loss = F.cross_entropy(
                mdl(x).reshape(-1, mdl.vocab_size),
                y.reshape(-1),
                ignore_index=0,
            ) / GRAD_ACCUM
        scaler.scale(loss).backward()
        loss_val += loss.item()

    scaler.unscale_(optimizer)
    gnorm = th.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    step += 1

    # ── Time limit → save last, not best ────────────────────
    elapsed_h = (time.time() - t0) / 3600
    if elapsed_h >= MAX_HOURS:
        last_path = SAVE_DIR / "pretrain_last.pt"
        th.save({"step": step, "loss": loss_val,
                 "model": _core(mdl).state_dict(), "optimizer": optimizer.state_dict()}, last_path)
        print(f"\n⏱ Time limit reached. Saved pretrain_last.pt (step={step:,})")
        print(f"Best checkpoint: pretrain_best.pt (eval={best_eval:.4f})")
        done = True
        break

    if step % LOG_EVERY == 0:
        tokens_m = step * batch_size * seq_len * GRAD_ACCUM / 1e6
        ppl      = math.exp(min(loss_val, 20))
        print(f"step={step:6d} | loss={loss_val:.4f} | ppl={ppl:.1f} | "
              f"lr={lr:.2e} | gnorm={gnorm:.2f} | "
              f"{elapsed_h:.2f}h | {tokens_m:.0f}M tok")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step, "loss": loss_val, "ppl": ppl,
                "lr": lr, "gnorm": float(gnorm),
                "elapsed_h": elapsed_h, "tokens_M": tokens_m,
            }) + "\n")

    if step % EVAL_EVERY == 0:
        ev = evaluate()
        ppl = math.exp(min(ev, 20))
        print(f"  ↳ eval={ev:.4f} | ppl={ppl:.1f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "eval_loss": ev}) + "\n")
        if ev < best_eval:
            best_eval = ev
            save_ckpt(step, ev)

        # ── Overfitting detection ───────────────────────────
        eval_history.append(ev)
        if len(eval_history) > OVERFIT_PATIENCE and all(
            eval_history[-i] > eval_history[-i - 1]
            for i in range(1, OVERFIT_PATIENCE + 1)
        ):
            print(f"\n⚠ OVERFITTING: eval loss rose {OVERFIT_PATIENCE}x in a row "
                  f"({eval_history[-OVERFIT_PATIENCE-1]:.4f} → {eval_history[-1]:.4f})")
            print(f"  Stopping early. Best: pretrain_best.pt (eval={best_eval:.4f})")
            done = True
            break

if not done:
    print(f"\nDone. step={step:,} | best_eval={best_eval:.4f}")
