import argparse, json, sys, subprocess, os
import torch as th

def _ensure_compatible_transformers():
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.36"):
            return
    except Exception:
        pass
    print("Downgrading transformers -> 4.35.2 (AraGPT2 needs transformers.onnx) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.35.2"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_compatible_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--qa", default="data/good_instruct_answers.jsonl")
    ap.add_argument("--base_model", default="aubmindlab/aragpt2-large")
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    try:
        from arabert.preprocess import ArabertPreprocessor
    except Exception:
        try:
            from arabert.arabert_preprocessor import ArabertPreprocessor
        except Exception:
            ArabertPreprocessor = None
    prep = None
    if ArabertPreprocessor is not None:
        try:
            prep = ArabertPreprocessor(model_name=args.base_model)
        except Exception as e:
            print(f"⚠ ArabertPreprocessor unavailable ({e}); running WITHOUT Arabic preprocessing")

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if th.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rows = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]

    for i, r in enumerate(rows, 1):
        q = r.get("instruction") or r.get("question") or r.get("q") or ""
        good = r.get("output") or r.get("answer") or r.get("a") or ""
        qc = prep.preprocess(q) if prep else q
        text = f"### Instruction:\n{qc}\n\n### Response:\n"
        ids = tok.encode(text, return_tensors="pt").to(device)
        with th.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                                 top_k=args.top_k, temperature=args.temperature,
                                 repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n===== [{i}/{len(rows)}] =====")
        print(f"Q: {q}")
        print(f"--- curated good answer ---\n{good}")
        print(f"--- model output ---\n{ans}")

if __name__ == "__main__":
    main()
