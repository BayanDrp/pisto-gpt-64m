# ============================================================
# Finetune a PRETRAINED Arabic model (AraGPT2-large) on our
# instruction data. Uses HuggingFace transformers + PyTorch.
# Structured with best practices from training/sft.py.
# Requires: torch, transformers, datasets, arabert, farasapy, bitsandbytes
# ============================================================
import sys, os, math, time, json, random, inspect, shutil
from pathlib import Path
import subprocess, pkg_resources

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Flush stdout per line so log viewers (Kaggle, etc.) show live progress
# instead of hiding everything behind a block buffer until it fills.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _ensure_compatible_transformers():
    # AraGPT2 custom code requires transformers.onnx, which was removed in transformers>=4.36
    try:
        import transformers
        ver = transformers.__version__
    except Exception:
        ver = "0"
    if ver == "0" or pkg_resources.parse_version(ver) >= pkg_resources.parse_version("4.36.0"):
        print("Downgrading transformers -> 4.35.2 (AraGPT2 needs transformers.onnx) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.35.2"])
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_compatible_transformers()

import torch as th
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Auto-answer any "Do you wish to run the custom code? [y/N]" prompt non-interactively
import builtins
builtins.input = lambda *a, **k: "y"

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
CONFIG_PATH = ROOT / "config" / os.getenv("HF_CONFIG", "finetune_hf.json")
CONFIG_DIR = CONFIG_PATH.parent

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

MODEL_NAME = cfg["model_name"]
TRUST_REMOTE = cfg.get("trust_remote_code", True)
MAX_LEN = cfg.get("max_len", 512)
USE_ARABERT = cfg.get("use_arabert", True)
train_cfg = cfg["training"]
ds_cfg = cfg["dataset"]
SAVE_DIR = (ROOT / cfg.get("save_dir", "weights_hf")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

random.seed(42)
th.manual_seed(42)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_gpus = th.cuda.device_count() if th.cuda.is_available() else 0
print(f"Device : {device} (gpus={n_gpus})")
if th.cuda.is_available():
    print(f"GPU    : {th.cuda.get_device_name(0)}")
print("finetune_hf.py | SFT Standard Architecture")

# ── Arabic Preprocessing (ArabertPreprocessor) ──────────────
prep = None
if USE_ARABERT:
    try:
        from arabert.preprocess import ArabertPreprocessor
    except Exception:
        try:
            from arabert.arabert_preprocessor import ArabertPreprocessor
        except Exception:
            ArabertPreprocessor = None
    if ArabertPreprocessor is not None:
        try:
            prep = ArabertPreprocessor(model_name=MODEL_NAME)
            print("ArabertPreprocessor ready ✓")
        except Exception as e:
            print(f"⚠ ArabertPreprocessor unavailable ({e}); running without preprocessing")


def clean(text):
    text = str(text).strip()
    if prep is not None and text:
        try:
            text = prep.preprocess(text)
        except Exception:
            pass
    return text


def format_prompt(instruction: str, inp: str = "") -> str:
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ── Load pretrained model & tokenizer ────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {MODEL_NAME} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=TRUST_REMOTE, torch_dtype=th.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id
EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"
VOCAB = model.config.vocab_size
print(f"Tokenizer vocab: {VOCAB}, pad_id={PAD_ID}, eos_token={EOS_TOKEN}")

model = model.to(device)
try:
    model.config.use_cache = False
except Exception:
    pass
grad_ckpt = train_cfg.get("gradient_checkpointing", True)
if grad_ckpt:
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled ✓")
    except Exception as e:
        print(f"⚠ gradient_checkpointing unavailable ({e})")
else:
    print("Gradient checkpointing DISABLED (faster, uses more memory)")

# Use every available GPU via DataParallel (unless single_gpu requested)
single_gpu = train_cfg.get("single_gpu", False)
using_dp = (n_gpus > 1) and not single_gpu
if using_dp:
    print(f"Using {n_gpus} GPUs via DataParallel")
    model = th.nn.DataParallel(model)
elif single_gpu and n_gpus > 1:
    th.cuda.set_device(0)
    print("Single-GPU mode (cuda:0) — DataParallel disabled by config")
elif n_gpus >= 1:
    print("Using single GPU (cuda:0)")

_core = model.module if isinstance(model, th.nn.DataParallel) else model
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ── Dataset extraction & Token Packing ───────────────────────
def _get_field(row, keys):
    for k in keys:
        if k in row and row[k]:
            return str(row[k])
    return ""


samples = []

# 1. Local curated data
local_path = (ROOT / ds_cfg.get("local_path", "data/good_instruct_answers.jsonl")).resolve()
if local_path.exists():
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            inst = clean(_get_field(obj, ["instruction", "prompt", "question"]))
            inp  = clean(_get_field(obj, ["input", "context"]))
            out  = clean(_get_field(obj, ["output", "response", "answer", "text"]))
            if inst and out:
                samples.append((inst, inp, out))
    print(f"Loaded {len(samples)} local Q&A pairs from {local_path.name}")
else:
    print(f"⚠ local data not found: {local_path}")

samples = samples * int(ds_cfg.get("local_repeat", 1))

# 2. Hugging Face datasets (real Arabic instruction corpora)
hf_names = ds_cfg.get("hf_datasets") or ([ds_cfg["hf_dataset"]] if ds_cfg.get("hf_dataset") else [])
hf_max = int(ds_cfg.get("hf_max", 4000))
hf_split = ds_cfg.get("hf_split", "train")


def _row_to_pair(row):
    """Return (instruction, input, output) or None for a HF dataset row."""
    if not isinstance(row, dict):
        return None
    keys = set(row.keys())
    if "instruction" in keys and ("output" in keys or "answer" in keys):
        inst = clean(row.get("instruction") or "")
        inp = clean(row.get("input") or "") if "input" in keys else ""
        out = clean(row.get("output") or "") or clean(row.get("answer") or "")
        return (inst, inp, out) if (inst and out) else None
    if "query" in keys and "answer" in keys:
        return (clean(row.get("query") or ""), "", clean(row.get("answer") or ""))
    if "prompt" in keys and "completion" in keys:
        return (clean(row.get("prompt") or ""), "", clean(row.get("completion") or ""))
    if "messages" in keys and isinstance(row.get("messages"), list):
        inst = out = None
        for m in row["messages"]:
            role = (m.get("role") or "").lower()
            content = clean(m.get("content") or "")
            if role == "user" and inst is None:
                inst = content
            elif role in ("assistant", "gpt") and out is None:
                out = content
        return (inst, "", out) if (inst and out) else None
    if "conversations" in keys and isinstance(row.get("conversations"), list):
        inst = out = None
        for m in row["conversations"]:
            frm = (m.get("from") or "").lower()
            val = clean(m.get("value") or "")
            if frm in ("human", "user") and inst is None:
                inst = val
            elif frm in ("gpt", "assistant") and out is None:
                out = val
        return (inst, "", out) if (inst and out) else None
    return None


total_hf = 0
for hf_name in hf_names:
    try:
        from datasets import load_dataset
        print(f"Loading HF dataset {hf_name} ...")
        load_kwargs = {"path": hf_name, "split": hf_split}
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN
        hf_ds = load_dataset(**load_kwargs)
        if isinstance(hf_ds, dict):
            hf_ds = hf_ds.get(hf_split) or next(iter(hf_ds.values()))
        added = 0
        for row in hf_ds:
            pair = _row_to_pair(row)
            if pair is None:
                continue
            inst, inp, out = pair
            if 1 < len(out) < 800 and len(inst) < 1200:
                samples.append((inst, inp, out))
                added += 1
            if hf_max and added >= hf_max:
                break
        total_hf += added
        print(f"Added {added} samples from HF dataset {hf_name} ✓")
    except Exception as e:
        print(f"⚠ dataset {hf_name} skipped ({e})")
if total_hf == 0:
    print("⚠ No HF dataset loaded; using local data only")
else:
    print(f"HF datasets contributed {total_hf:,} samples")

if not samples:
    raise SystemExit("No training samples found — aborting.")

random.shuffle(samples)
print(f"Total training samples: {len(samples):,}")


class InstructDataset(Dataset):
    """Flat packed-token dataset with EOS tokens and safe chunking."""
    def __init__(self, sample_list, seq_len=MAX_LEN):
        self.seq_len = seq_len
        toks = []
        for inst, inp, resp in sample_list:
            text = format_prompt(inst, inp) + resp + EOS_TOKEN
            encoded = tokenizer.encode(text, add_special_tokens=False)
            toks.extend(encoded)
        self.data = th.tensor(toks, dtype=th.long)
        self.n = max(1, (len(self.data) - 1) // seq_len)
        if len(self.data) < seq_len + 1:
            pad = th.full((seq_len + 1 - len(self.data),), PAD_ID, dtype=th.long)
            self.data = th.cat([self.data, pad])
        print(f"Tokens: {len(toks):,} → {self.n:,} chunks (seq_len={seq_len})")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.seq_len
        chunk = self.data[s:s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = th.cat([chunk, th.full((self.seq_len + 1 - len(chunk),), PAD_ID, dtype=th.long)])
        return chunk


def collate(batch):
    return th.stack(batch)


split = int(len(samples) * ds_cfg.get("train_split", 0.95))
train_ds = InstructDataset(samples[:split], seq_len=MAX_LEN)
eval_ds  = InstructDataset(samples[split:], seq_len=MAX_LEN)

batch_size = max(1, train_cfg.get("batch_size", 1))
if using_dp and batch_size < n_gpus:
    # give each GPU >=1 sample so DataParallel is actually effective
    batch_size = n_gpus
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
print("DataLoaders ready ✓")

# ── Optimizer & Scheduler ────────────────────────────────────
MAX_HOURS       = train_cfg.get("max_hours", 2.0)
GRAD_ACCUM      = train_cfg.get("grad_accum", 8)
MAX_STEPS       = train_cfg.get("max_steps", 3000)
WARMUP          = train_cfg.get("warmup_steps", 100)
LOG_EVERY       = train_cfg.get("log_every", 10)
EVAL_EVERY      = train_cfg.get("eval_every", 150)
LR              = train_cfg.get("lr", 2e-5)
LR_MIN          = train_cfg.get("lr_min", 2e-6)
LABEL_SMOOTHING = train_cfg.get("label_smoothing", 0.0)
OVERFIT_PAT     = train_cfg.get("overfit_patience", 5)

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() < 2 or any(x in name for x in ["ln", "bias", "embed", "LayerNorm"]):
        no_decay.append(p)
    else:
        decay.append(p)


def _make_optimizer(decay, no_decay):
    try:
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit([
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=LR, betas=(0.9, 0.95)), True
    except Exception:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit([
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=LR, betas=(0.9, 0.95)), True
    except Exception as e:
        print(f"⚠ bitsandbytes unavailable ({e}); using standard AdamW")
        return th.optim.AdamW([
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=LR, betas=(0.9, 0.95)), False


optimizer, _use_8bit = _make_optimizer(decay, no_decay)
print("Using 8-bit AdamW (bitsandbytes) ✓" if _use_8bit else "Using standard AdamW")
scaler = GradScaler("cuda", enabled=device.type == "cuda")


def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / max(WARMUP, 1)
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return LR_MIN + (LR - LR_MIN) * cosine


# ── Evaluator (Loss + Token Accuracy) ────────────────────────
@th.no_grad()
def evaluate(mdl, loader, steps=30):
    mdl.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = 0
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        # Evaluated in fp32 for numerical stability
        logits = mdl(x).logits
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB),
            y.reshape(-1),
            ignore_index=PAD_ID,
            label_smoothing=LABEL_SMOOTHING,
        )
        if not th.isfinite(loss):
            continue
        total_loss += loss.item()
        pred = logits.argmax(-1)
        mask = y != PAD_ID
        total_correct += (pred[mask] == y[mask]).float().sum().item()
        total_tokens += mask.sum().item()
        n += 1
    mdl.train()
    return total_loss / max(n, 1), total_correct / max(total_tokens, 1)


def _copy_custom_code(dst):
    copied = []
    for target in [_core, _core.config]:
        try:
            mod = sys.modules[target.__class__.__module__]
            f = inspect.getfile(mod)
            if os.path.exists(f):
                shutil.copy(f, os.path.join(dst, os.path.basename(f)))
                copied.append(os.path.basename(f))
        except Exception:
            pass
    if copied:
        print(f"  ✓ copied custom code: {copied}")


def save_checkpoint(step, loss, name="best"):
    save_path = SAVE_DIR if name == "best" else SAVE_DIR / f"checkpoint_{name}"
    save_path.mkdir(parents=True, exist_ok=True)
    _core.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    _copy_custom_code(str(save_path))
    th.save({"step": step, "loss": loss}, save_path / "train_state.pt")
    print(f"  ✓ saved [{name}] model -> {save_path} (step={step:,} loss={loss:.4f})")


# ── Training Loop ────────────────────────────────────────────
step = 0
best_eval = float("inf")
bad_evals = 0
done = False
t0 = time.time()
model.train()
log_path = SAVE_DIR / "finetune_log.jsonl"

print(f"\n{'='*50}")
print(f"Finetune {MODEL_NAME} | max {MAX_HOURS}h | max_steps={MAX_STEPS} | lr={LR}")
print(f"{'='*50}\n")

loader_iter = iter(train_loader)

while step < MAX_STEPS and not done:
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for _ in range(GRAD_ACCUM):
        try:
            micro = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            micro = next(loader_iter)

        x = micro[:, :-1].to(device, non_blocking=True)
        y = micro[:, 1:].to(device, non_blocking=True)

        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x).logits
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                y.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=LABEL_SMOOTHING,
            ) / GRAD_ACCUM

        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(optimizer)
    gnorm = th.nn.utils.clip_grad_norm_(_core.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    step += 1

    elapsed_h = (time.time() - t0) / 3600.0
    if elapsed_h >= MAX_HOURS:
        save_checkpoint(step, loss_accum, name="last")
        print(f"\n⏱ Time limit reached ({elapsed_h:.2f}h).")
        done = True
        break

    if step % LOG_EVERY == 0:
        ppl = math.exp(min(loss_accum, 20))
        print(f"step={step:5d} | loss={loss_accum:.4f} | ppl={ppl:.1f} | lr={lr:.2e} | gnorm={gnorm:.2f} | {elapsed_h:.2f}h", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step, "loss": loss_accum, "ppl": ppl, "lr": lr,
                "gnorm": float(gnorm), "elapsed_h": elapsed_h
            }) + "\n")

    if step % EVAL_EVERY == 0:
        el, acc = evaluate(model, eval_loader)
        ppl = math.exp(min(el, 20))
        print(f"  ↳ [eval] step={step} | eval_loss={el:.4f} | token_acc={acc:.3f} | ppl={ppl:.1f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "eval_loss": el, "token_acc": acc}) + "\n")

        if math.isfinite(el) and el < best_eval:
            best_eval = el
            bad_evals = 0
            save_checkpoint(step, el, name="best")
        else:
            bad_evals += 1
            if bad_evals >= OVERFIT_PAT:
                print(f"\n⚠ Early stopping: eval loss increased {bad_evals}x in a row. Best eval={best_eval:.4f}")
                done = True
                break

if not done:
    save_checkpoint(step, best_eval, name="last")
    print(f"\nTraining Complete. step={step:,} | best_eval={best_eval:.4f}")

