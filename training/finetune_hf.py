# ============================================================
# Finetune a PRETRAINED Arabic model (AraGPT2-large) on our
# instruction data. Uses HuggingFace transformers + PyTorch.
# The borrowed model already speaks Arabic; we only adapt it.
# Requires: torch, transformers, datasets, arabert, farasapy
# ============================================================
import sys, os, math, time, json, random
from pathlib import Path

import subprocess, pkg_resources
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure_compatible_transformers():
    # AraGPT2's custom model code imports transformers.onnx, which was
    # removed in transformers>=4.36. Auto-downgrade so it just works.
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
from torch.utils.data import Dataset, DataLoader

_HERE       = Path(__file__).parent
ROOT        = _HERE.parent
CONFIG_PATH = ROOT / "config" / "finetune_hf.json"
CONFIG_DIR  = CONFIG_PATH.parent

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

MODEL_NAME   = cfg["model_name"]
TRUST_REMOTE = cfg["trust_remote_code"]
MAX_LEN      = cfg["max_len"]
USE_ARABERT  = cfg["use_arabert"]
train_cfg    = cfg["training"]
ds_cfg       = cfg["dataset"]
SAVE_DIR     = (ROOT / cfg["save_dir"]).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device : {device}")
if th.cuda.is_available():
    print(f"GPU    : {th.cuda.get_device_name(0)}")

# ── Arabic preprocessing (REQUIRED for AraGPT2) ─────────────
prep = None
if USE_ARABERT:
    try:
        from arabert.preprocess import ArabertPreprocessor
    except Exception:
        from arabert.arabert_preprocessor import ArabertPreprocessor
    try:
        prep = ArabertPreprocessor(model_name=MODEL_NAME)
        print("ArabertPreprocessor ready ✓")
    except Exception as e:
        print(f"⚠ ArabertPreprocessor unavailable ({e}); continuing WITHOUT Arabic preprocessing")

def clean(text):
    text = str(text).strip()
    if prep is not None:
        try:
            text = prep.preprocess(text)
        except Exception:
            pass
    return text

# ── Load pretrained model + tokenizer ───────────────────────
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
print(f"Loading {MODEL_NAME} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=TRUST_REMOTE, torch_dtype=th.float32)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id
VOCAB  = model.config.vocab_size
print(f"Tokenizer vocab: {VOCAB}, pad_id={PAD_ID}")

model = model.to(device)
try:
    model.config.use_cache = False
except Exception:
    pass
try:
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled ✓")
except Exception as e:
    print(f"⚠ gradient_checkpointing unavailable ({e})")

if th.cuda.is_available() and th.cuda.device_count() > 1:
    print(f"Using {th.cuda.device_count()} GPUs via DataParallel")
    model = th.nn.DataParallel(model)
_core = model.module if isinstance(model, th.nn.DataParallel) else model
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── Build dataset ───────────────────────────────────────────
PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
samples = []

local_path = (ROOT / ds_cfg["local_path"]).resolve()
if local_path.exists():
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inst, out = clean(obj.get("instruction", "")), clean(obj.get("output", ""))
            if inst and out:
                samples.append((inst, out))
    print(f"Loaded {len(samples)} local Q&A from {local_path}")
else:
    print(f"⚠ local data not found: {local_path}")

samples = samples * int(ds_cfg.get("local_repeat", 1))

candidates = []
if ds_cfg.get("hf_dataset"):
    candidates.append(ds_cfg["hf_dataset"])
candidates += ["arbml/ALPACA_AR", "M4ali/arabic-instruct", "OALL/Alpaca-Arabic", "vineetsharma/arabic-instruct"]
hf_loaded = False
for hf_name in candidates:
    try:
        from datasets import load_dataset
        print(f"Loading HF dataset {hf_name} ...")
        load_kwargs = {"path": hf_name, "split": ds_cfg.get("hf_split", "train")}
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN
        hf_ds = load_dataset(**load_kwargs)
        inst_f, out_f = ds_cfg.get("hf_instruction_field", "instruction"), ds_cfg.get("hf_output_field", "output")
        added = 0
        for row in hf_ds:
            inst, out = clean(row.get(inst_f, "") or ""), clean(row.get(out_f, "") or "")
            if inst and out and len(out) < 600:
                samples.append((inst, out)); added += 1
            if ds_cfg.get("hf_max") and added >= ds_cfg["hf_max"]:
                break
        print(f"Added {added} samples from HF dataset {hf_name}")
        hf_loaded = True
        break
    except Exception as e:
        print(f"⚠ dataset {hf_name} unavailable ({e}); trying next")
if not hf_loaded:
    print("⚠ No HF dataset loaded; using local data only")

if not samples:
    raise SystemExit("No training samples found — aborting.")
random.shuffle(samples)
print(f"Total training samples: {len(samples):,}")

class InstructDataset(Dataset):
    def __init__(self, samples, seq_len=MAX_LEN):
        self.seq_len = seq_len
        print(f"Tokenizing {len(samples):,} samples ...")
        toks = []
        for inst, resp in samples:
            text = PROMPT.format(instruction=inst, response=resp)
            toks.extend(tokenizer.encode(text, add_special_tokens=False))
        self.data = th.tensor(toks, dtype=th.long)
        self.n = (len(self.data) - 1) // seq_len
        if self.n == 0:
            raise ValueError(f"Not enough tokens: {len(self.data)} < {seq_len+1}")
        print(f"Tokens: {len(self.data):,} → {self.n:,} chunks")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s:s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = th.cat([chunk, th.full((self.seq_len + 1 - len(chunk),), PAD_ID, dtype=th.long)])
        return chunk

split     = int(len(samples) * ds_cfg.get("train_split", 0.95))
train_ds  = InstructDataset(samples[:split])
eval_ds   = InstructDataset(samples[split:])
batch_size = train_cfg["batch_size"]
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=th.cuda.is_available(), drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=th.cuda.is_available(), drop_last=True)
print("Dataset ready ✓")

# ── Training setup ──────────────────────────────────────────
MAX_HOURS  = train_cfg["max_hours"]
GRAD_ACCUM = train_cfg["grad_accum"]
MAX_STEPS  = train_cfg["max_steps"]
WARMUP     = train_cfg["warmup_steps"]
LOG_EVERY  = train_cfg["log_every"]
EVAL_EVERY = train_cfg["eval_every"]
LR         = train_cfg["lr"]
LR_MIN     = train_cfg["lr_min"]
OVERFIT_PAT= train_cfg.get("overfit_patience", 5)

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() < 2 or any(x in name for x in ["ln", "bias", "embed", "LayerNorm"]):
        no_decay.append(p)
    else:
        decay.append(p)

try:
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.95))
    print("Using 8-bit AdamW (bitsandbytes) ✓")
except Exception as e:
    print(f"⚠ bitsandbytes unavailable ({e}); using standard AdamW")
    optimizer = th.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.95), fused=th.cuda.is_available())
scaler = th.amp.GradScaler("cuda", enabled=th.cuda.is_available())

def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return LR_MIN + (LR - LR_MIN) * cosine

@th.no_grad()
def evaluate(n=30):
    model.eval(); total = 0.0; nb = 0
    try:
        for i, batch in enumerate(eval_loader):
            if i >= n:
                break
            x = batch[:, :-1].to(device); y = batch[:, 1:].to(device)
            with th.amp.autocast("cuda", enabled=th.cuda.is_available()):
                loss = F.cross_entropy(model(x).logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=PAD_ID)
            total += loss.item(); nb += 1
    finally:
        model.train()
    return total / max(nb, 1)

def save_best(step, loss):
    _core.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    th.save({"step": step, "loss": loss}, SAVE_DIR / "train_state.pt")
    print(f"  ✓ saved finetuned model -> {SAVE_DIR} (step={step:,})")

step = 0; best_eval = float("inf"); eval_history = []; done = False
t0 = time.time(); model.train()
log_path = SAVE_DIR / "finetune_log.jsonl"
print(f"\n{'='*50}\nFinetune {MODEL_NAME} | {MAX_HOURS}h | lr={LR}\n{'='*50}\n")
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
            loss = F.cross_entropy(model(x).logits.reshape(-1, VOCAB), y.reshape(-1),
                                   ignore_index=PAD_ID) / GRAD_ACCUM
        scaler.scale(loss).backward()
        loss_val += loss.item()
    scaler.unscale_(optimizer)
    gnorm = th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update()
    step += 1

    elapsed = (time.time() - t0) / 3600
    if elapsed >= MAX_HOURS:
        _core.save_pretrained(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
        th.save({"step": step, "loss": loss_val}, SAVE_DIR / "train_state.pt")
        print(f"\n⏱ Time limit. Saved -> {SAVE_DIR} (step={step:,})")
        done = True; break

    if step % LOG_EVERY == 0:
        ppl = math.exp(min(loss_val, 20))
        print(f"step={step:5d} | loss={loss_val:.4f} | ppl={ppl:.1f} | lr={lr:.2e} | gnorm={gnorm:.2f} | {elapsed:.2f}h")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": loss_val, "ppl": ppl, "lr": lr, "gnorm": float(gnorm), "elapsed_h": elapsed}) + "\n")

    if step % EVAL_EVERY == 0:
        ev = evaluate(); ppl = math.exp(min(ev, 20))
        print(f"  ↳ eval={ev:.4f} | ppl={ppl:.1f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "eval_loss": ev}) + "\n")
        if ev < best_eval:
            best_eval = ev; save_best(step, ev)
        eval_history.append(ev)
        if len(eval_history) > OVERFIT_PAT and all(eval_history[-i] > eval_history[-i-1] for i in range(1, OVERFIT_PAT+1)):
            print(f"\n⚠ OVERFITTING {OVERFIT_PAT}x — stopping. Best eval={best_eval:.4f}")
            done = True; break

if not done:
    print(f"\nDone. step={step:,} | best_eval={best_eval:.4f}")
