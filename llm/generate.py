import json
import torch as th
import torch.nn.functional as F
from pathlib import Path
from model import Model

# ── Load config ──────────────────────────────────────────────
_HERE = Path(__file__).parent
CONFIG_PATH = _HERE.parent / "config" / "generate.json"
CONFIG_DIR  = CONFIG_PATH.parent

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

model_cfg = cfg["model"]
gen_cfg   = cfg["generation"]
weight_path_default = str((CONFIG_DIR / cfg["weight_path"]).resolve())

device = "cuda" if th.cuda.is_available() else "cpu"

_loaded_models = {}


def load_model(weight_path=None):
    weight_path = weight_path or weight_path_default
    if weight_path in _loaded_models:
        return _loaded_models[weight_path]

    mdl = Model(
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        transformer_layers=model_cfg["transformer_layers"],
        max_len=model_cfg["max_len"],
    ).to(device)
    mdl.lm_head.weight = mdl.embed.weight

    ckpt = th.load(weight_path, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    mdl.eval()
    print(f"Loaded ✓ | {weight_path} | step={ckpt['step']:,} | loss={ckpt['loss']:.4f}")

    _loaded_models[weight_path] = mdl
    return mdl


def chat(
    prompt,
    weight_path=None,
    max_new=None,
    temp=None,
    top_k=None,
    top_p=None,
    rep_pen=None,
    inp="",
):
    # Fall back to config values if not provided
    max_new  = max_new  if max_new  is not None else gen_cfg["max_new"]
    temp     = temp     if temp     is not None else gen_cfg["temp"]
    top_k    = top_k    if top_k    is not None else gen_cfg["top_k"]
    top_p    = top_p    if top_p    is not None else gen_cfg["top_p"]
    rep_pen  = rep_pen  if rep_pen  is not None else gen_cfg["rep_pen"]

    mdl = load_model(weight_path)

    if inp:
        formatted = f"### Instruction:\n{prompt}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    tokens = mdl.tokenizer.tokenize(formatted)
    if tokens[-1] == mdl.tokenizer.eos_id:
        tokens = tokens[:-1]

    input_ids = th.tensor([tokens], dtype=th.long, device=device)
    generated = []

    with th.no_grad():
        for _ in range(max_new):
            logits = mdl(input_ids)[:, -1, :]

            if rep_pen != 1.0 and generated:
                for tid in set(generated):
                    logits[0, tid] /= rep_pen if logits[0, tid] > 0 else 1 / rep_pen

            logits = logits / temp

            if top_k > 0:
                top_vals, _ = th.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sl, si = th.sort(logits, descending=True)
                cum = th.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum - F.softmax(sl, dim=-1) >= top_p] = float("-inf")
                logits = th.zeros_like(logits).scatter_(1, si, sl)

            next_id = th.multinomial(F.softmax(logits, dim=-1), 1).item()
            if next_id == mdl.tokenizer.eos_id:
                break

            generated.append(next_id)
            input_ids = th.cat([input_ids, th.tensor([[next_id]], device=device)], dim=1)

            # Stop if the model begins a new instruction turn instead of
            # ending its response (small models sometimes chain examples).
            if len(generated) >= 6:
                tail = mdl.tokenizer.detokenize(generated[-12:])
                if "### Instruction:" in tail or "## Instruction:" in tail:
                    break

    return mdl.tokenizer.detokenize(generated).strip()


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Tell me a short story."
    ]

    print("\n" + "=" * 50)
    for q in tests:
        print(f"Q: {q}")
        print(f"A: {chat(q)}")
        print("-" * 35)