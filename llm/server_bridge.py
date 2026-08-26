#!/usr/bin/env python3
"""Bridge between the Go web server and the language model.

Reads ONE JSON object from stdin:
    {"prompt": str, "max_new": int, "temp": float, "top_k": int, "top_p": float}
Only "prompt" is required; the rest fall back to config/generate.json.

Prints ONE JSON object to stdout:
    {"response": "<generated text>"}   or   {"error": "<message>"}

Model selection (drop-in first, so "put weights in weights/ and run" works):
  1. <repo>/weights/  if it holds a HuggingFace model (config.json or *.safetensors/*.bin)
  2. config/generate.json "weight_path":
       - a directory with HF files  -> HuggingFace model
       - a *.pt file                 -> the original 68M transformer
"""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

# Make llm/ importable no matter what CWD the Go server uses.
_LLM_DIR = Path(__file__).parent
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))
_ROOT = _LLM_DIR.parent


def _ensure_compatible_transformers():
    """AraGPT2's custom code imports transformers.onnx (removed in >=4.36).

    If a too-new transformers is present, try to pin 4.35.2 and re-exec.
    Failures are non-fatal: the legacy .pt path doesn't need transformers,
    and the HF path will surface a clear error if it's truly missing.
    """
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.36"):
            return
    except Exception:
        return
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "transformers==4.35.2"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(
            f"[bridge] warning: could not pin transformers==4.35.2 ({e}); "
            "HF model may fail to load",
            flush=True,
        )


_ensure_compatible_transformers()

import torch as th

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# Cache the loaded HuggingFace model across calls within one process
# (the Go server spawns a fresh process per request, but keep it simple).
_HF = None
_HF_DIR = None

GEN_DEFAULTS = {"max_new": 100, "temp": 0.5, "top_k": 20, "top_p": 0.9, "rep_pen": 1.3}


def _load_gen_defaults():
    try:
        cfg = json.loads((_ROOT / "config" / "generate.json").read_text(encoding="utf-8"))
        g = cfg.get("generation", {})
        for k in ("max_new", "temp", "top_k", "top_p", "rep_pen"):
            if k in g:
                GEN_DEFAULTS[k] = g[k]
    except Exception:
        pass


def _hf_dir(d: Path):
    """Return the HF model directory under d (or d itself), or None."""
    if (d / "config.json").is_file() or any(
        p.suffix in (".safetensors", ".bin") for p in d.iterdir()
    ):
        return d
    for sub in d.iterdir():
        if sub.is_dir():
            found = _hf_dir(sub)
            if found is not None:
                return found
    return None


def _resolve_model():
    """Return (path_or_dir, kind) where kind is 'hf' or 'pt', else (None, None)."""
    # 1) drop-in weights/ folder (priority) — root or a subfolder
    drop = _ROOT / "weights"
    if drop.is_dir():
        hf = _hf_dir(drop)
        if hf is not None:
            return str(hf), "hf"

    # 2) config/generate.json weight_path
    cfg_path = _ROOT / "config" / "generate.json"
    if cfg_path.is_file():
        try:
            wp = json.loads(cfg_path.read_text(encoding="utf-8")).get("weight_path")
            if wp:
                p = (cfg_path.parent / wp).resolve()
                if p.is_dir() and _hf_dir(p) is not None:
                    return str(p), "hf"
                if p.suffix == ".pt":
                    return str(p), "pt"
        except Exception:
            pass
    return None, None


def _load_hf(model_dir):
    global _HF, _HF_DIR
    if _HF is not None:
        return _HF
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(DEVICE).eval()

    prep = None
    try:
        from arabert.preprocess import ArabertPreprocessor
    except Exception:
        try:
            from arabert.arabert_preprocessor import ArabertPreprocessor
        except Exception:
            ArabertPreprocessor = None
    if ArabertPreprocessor is not None:
        try:
            prep = ArabertPreprocessor(model_name="aubmindlab/aragpt2-large")
        except Exception:
            prep = None

    _HF = (model, tok, prep)
    _HF_DIR = model_dir
    return _HF


def chat_hf(prompt, max_new, temp, top_k, top_p, rep_pen):
    model, tok, prep = _load_hf(_HF_DIR)
    q = prep.preprocess(prompt) if (prep is not None and prompt) else prompt
    text = f"### Instruction:\n{q}\n\n### Response:\n"
    ids = tok.encode(text, return_tensors="pt").to(DEVICE)
    with th.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=int(max_new),
            do_sample=True,
            temperature=float(temp),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(rep_pen),
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    if prep is not None and hasattr(prep, "desegment"):
        try:
            ans = prep.desegment(ans)
        except Exception:
            pass
    return ans


def _coerce(req, key, cast):
    if key not in req or req[key] is None:
        return None
    try:
        return cast(req[key])
    except (TypeError, ValueError):
        raise ValueError(f"invalid value for '{key}': {req[key]!r}")


def main():
    _load_gen_defaults()
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise ValueError("empty request on stdin")
        req = json.loads(raw)
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")

        prompt = req.get("prompt")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is empty")

        max_new = _coerce(req, "max_new", int) or GEN_DEFAULTS["max_new"]
        temp = _coerce(req, "temp", float) or GEN_DEFAULTS["temp"]
        top_k = _coerce(req, "top_k", int) or GEN_DEFAULTS["top_k"]
        top_p = _coerce(req, "top_p", float) or GEN_DEFAULTS["top_p"]
        rep_pen = _coerce(req, "rep_pen", float) or GEN_DEFAULTS["rep_pen"]

        target, kind = _resolve_model()
        if target is None:
            raise RuntimeError(
                "No model found. Drop HuggingFace weights into weights/ "
                "(model.safetensors + config.json + tokenizer files + custom code) "
                "or set config/generate.json 'weight_path'."
            )

        if kind == "hf":
            global _HF_DIR
            _HF_DIR = target
            with contextlib.redirect_stdout(io.StringIO()):
                text = chat_hf(prompt, max_new, temp, top_k, top_p, rep_pen)
        else:  # legacy 68M transformer (.pt)
            from generate import chat
            with contextlib.redirect_stdout(io.StringIO()):
                text = chat(
                    prompt=str(prompt),
                    weight_path=target,
                    max_new=max_new,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    rep_pen=rep_pen,
                )

        print(json.dumps({"response": text}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the server
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
