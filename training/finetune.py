# ============================================================
# Instruction Tuning — loads all settings from config/instruct.json
# ============================================================
import sys, os, math, time, json, random
from pathlib import Path

import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Load config ──────────────────────────────────────────────
_HERE       = Path(__file__).parent
CONFIG_PATH = _HERE.parent / "config" / "instruct.json"
CONFIG_DIR  = CONFIG_PATH.parent

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

model_cfg   = cfg["model"]
train_cfg   = cfg["training"]
dataset_cfg = cfg["dataset"]
SAVE_DIR    = (CONFIG_DIR / cfg["save_dir"]).resolve()
PRETRAINED  = (CONFIG_DIR / cfg["pretrained_weights"]).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN    = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ── Model import ─────────────────────────────────────────────
sys.path.insert(0, str(_HERE.parent / "llm"))
from model import Model as LLMModel
from tokenizer import ByteTokenizer

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device : {device}")
if th.cuda.is_available():
    print(f"GPU    : {th.cuda.get_device_name(0)}")
print(f"Config : {CONFIG_PATH}")
print(f"Save   : {SAVE_DIR}")

# ── Model + load pretrained ──────────────────────────────────
mdl = LLMModel(
    d_model=model_cfg["d_model"],
    nhead=model_cfg["nhead"],
    dim_feedforward=model_cfg["dim_feedforward"],
    dropout=model_cfg["dropout"],
    transformer_layers=model_cfg["transformer_layers"],
    max_len=model_cfg["max_len"],
).to(device)
mdl.lm_head.weight = mdl.embed.weight

if PRETRAINED.exists():
    ckpt = th.load(PRETRAINED, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    print(f"Loaded pretrained ✓ | step={ckpt.get('step', 0):,}")
else:
    print(f"⚠ Pretrained weights not found at {PRETRAINED} — starting from scratch")

print(f"Parameters: {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M")

# ── Dataset ──────────────────────────────────────────────────
from datasets import load_dataset

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

MANUAL_DATA_PATH = (CONFIG_DIR / cfg["manual_data_path"]).resolve()

# Load manual Q&A pairs from data folder
with open(MANUAL_DATA_PATH) as f:
    manual_data = [tuple(pair) for pair in json.load(f)]
print(f"Loaded {len(manual_data)} manual Q&A pairs from {MANUAL_DATA_PATH}")

# Alpaca dataset (filtered)
print(f"Loading {dataset_cfg['alpaca_name']}...")
load_kwargs = {
    "path": dataset_cfg["alpaca_name"],
    "split": dataset_cfg["alpaca_split"],
}
if HF_TOKEN:
    load_kwargs["token"] = HF_TOKEN
ds = load_dataset(**load_kwargs)
alpaca = [
    (s["instruction"], s["output"])
    for s in ds
    if not s.get("input", "")
    and len(s["output"])      < dataset_cfg["alpaca_max_output_len"]
    and len(s["instruction"]) < dataset_cfg["alpaca_max_instruction_len"]
    and len(s["output"])      > dataset_cfg["alpaca_min_output_len"]
]
random.shuffle(alpaca)
alpaca = alpaca[: dataset_cfg["alpaca_max"]]
print(f"Alpaca filtered: {len(alpaca):,}")

all_data = manual_data * dataset_cfg["manual_repeat"] + alpaca
random.shuffle(all_data)
print(f"Total: {len(all_data):,} samples")

# ── InstructDataset ──────────────────────────────────────────
class InstructDataset(Dataset):
    def __init__(self, samples, tokenizer, seq_len=512):
        self.seq_len = seq_len
        print(f"Tokenizing {len(samples):,} samples...")
        all_tokens = []
        for inst, resp in samples:
            text = PROMPT.format(instruction=inst, response=resp)
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

tokenizer  = ByteTokenizer()
seq_len    = model_cfg["max_len"]
batch_size = train_cfg["batch_size"]

split    = int(len(all_data) * dataset_cfg["train_split"])
train_ds = InstructDataset(all_data[:split], tokenizer, seq_len=seq_len)
eval_ds  = InstructDataset(all_data[split:], tokenizer, seq_len=seq_len)

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
    {"params": decay,    "weight_decay": 0.01},
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
                    mdl(x).reshape(-1, mdl.vocab_size),
                    y.reshape(-1),
                    ignore_index=0,
                )
            total += loss.item()
            n_batches += 1
    finally:
        mdl.train()
    return total / max(n_batches, 1)

def save_ckpt(step, loss):
    path = SAVE_DIR / "instruct_best.pt"
    th.save({
        "step": step,
        "loss": loss,
        "model": mdl.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ instruct_best.pt saved (step={step:,})")

# ── Resume ───────────────────────────────────────────────────
step      = 0
best_eval = float("inf")
loss_val  = 0.0

resume_path = SAVE_DIR / "instruct_best.pt"
if resume_path.exists():
    print(f"Resuming from {resume_path}...")
    ckpt = th.load(resume_path, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    best_eval = ckpt.get("loss", float("inf"))
    print(f"Resumed instruct: step={step:,} | best_eval={best_eval:.4f}")
else:
    print("Starting Instruction Tuning fresh")

# ── Training loop ────────────────────────────────────────────
log_path = SAVE_DIR / "instruct_log.jsonl"
t0       = time.time()
mdl.train()

print(f"\n{'='*50}")
print(f"Instruction Tuning | max {MAX_HOURS}h | lr={LR} | overfit_patience={OVERFIT_PATIENCE}")
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
        last_path = SAVE_DIR / "instruct_last.pt"
        th.save({"step": step, "loss": loss_val,
                 "model": mdl.state_dict(), "optimizer": optimizer.state_dict()}, last_path)
        print(f"\n⏱ Time limit reached. Saved instruct_last.pt (step={step:,})")
        print(f"Best checkpoint: instruct_best.pt (eval={best_eval:.4f})")
        done = True
        break

    if step % LOG_EVERY == 0:
        ppl = math.exp(min(loss_val, 20))
        print(f"step={step:5d} | loss={loss_val:.4f} | ppl={ppl:.1f} | "
              f"lr={lr:.2e} | gnorm={gnorm:.2f} | {elapsed_h:.2f}h")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step, "loss": loss_val, "ppl": ppl,
                "lr": lr, "gnorm": float(gnorm),
                "elapsed_h": elapsed_h,
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
            print(f"  Stopping early. Best: instruct_best.pt (eval={best_eval:.4f})")
            done = True
            break

if not done:
    print(f"\nDone. step={step:,} | best_eval={best_eval:.4f}")
