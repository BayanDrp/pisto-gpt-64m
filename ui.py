import argparse
import os
import sys
import subprocess
import torch as th
import gradio as gr


def _ensure_compatible_transformers():
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse("4.36"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "transformers==4.35.2"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "transformers==4.35.2"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_compatible_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "aubmindlab/aragpt2-large"
MODEL_DIR_OVERRIDE = None
MODEL = None
TOK = None
PREP = None
PAD_ID = None
EOS_ID = None
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


def find_model_dir():
    if MODEL_DIR_OVERRIDE:
        return MODEL_DIR_OVERRIDE
    for d in ["weights", "weights_hf"]:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            return d
    if os.path.isdir("weights") and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir("weights")
    ):
        return "weights"
    return "weights"


def format_prompt(instruction, inp=""):
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load():
    global MODEL, TOK, PREP, PAD_ID, EOS_ID
    model_dir = find_model_dir()
    print(f"[ui] loading model from: {model_dir}", flush=True)
    try:
        MODEL = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"[ui] trust_remote_code=True failed ({e}); retrying without", flush=True)
        MODEL = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=False)
    MODEL.to(DEVICE).eval()
    try:
        TOK = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        TOK = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    PAD_ID = TOK.pad_token_id
    EOS_ID = TOK.eos_token_id

    PREP = None
    try:
        from arabert.preprocess import ArabertPreprocessor
    except Exception:
        try:
            from arabert.arabert_preprocessor import ArabertPreprocessor
        except Exception:
            ArabertPreprocessor = None
    if ArabertPreprocessor is not None:
        try:
            PREP = ArabertPreprocessor(model_name=BASE_MODEL)
        except Exception as e:
            print(f"[ui] ArabertPreprocessor unavailable ({e}); skipping", flush=True)


def respond(message, history, temperature, top_p, max_new, rep_penalty):
    if MODEL is None or TOK is None:
        return (
            "⚠ No model loaded. Drop the weights into the `weights/` folder "
            "(model.safetensors + config.json + tokenizer files + custom code) "
            "and restart `python ui.py`."
        )
    q = PREP.preprocess(message) if (PREP is not None and message) else message
    text = format_prompt(q)
    ids = TOK.encode(text, return_tensors="pt").to(DEVICE)
    with th.no_grad():
        out = MODEL.generate(
            ids,
            max_new_tokens=int(max_new),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(rep_penalty),
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
        )
    ans = TOK.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    if PREP is not None and hasattr(PREP, "desegment"):
        try:
            ans = PREP.desegment(ans)
        except Exception:
            pass
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="override auto-detected weights dir")
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    global MODEL_DIR_OVERRIDE
    MODEL_DIR_OVERRIDE = args.model_dir
    try:
        load()
    except Exception as e:
        print(f"[ui] model load failed: {e}", flush=True)
        MODEL = None

    demo = gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
            gr.Slider(32, 256, value=120, step=8, label="Max new tokens"),
            gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Repetition penalty"),
        ],
        title="AraGPT2 Fine-tuned Chat",
        description="Drop weights into `weights/` and chat. Arabic instruction model "
                    "(finetuned AraGPT2-large).",
        examples=[
            "ما عاصمة الأردن؟",
            "اكتب قصة قصيرة عن قطة ضاعت في المدينة.",
            "اشرح لي مفهوم الصبر في جملة واحدة.",
        ],
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
