import argparse, json, os, sys, subprocess, pkg_resources
import torch as th

def _ensure_compatible_transformers():
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

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "ما عاصمة الأردن؟",
    "ما عاصمة المغرب؟",
    "ما عاصمة العراق؟",
    "ما عاصمة سوريا؟",
    "ما عاصمة تركيا؟",
    "ما عاصمة ألمانيا؟",
    "ما عاصمة اليابان؟",
    "ما عاصمة إيطاليا؟",
    "ما عاصمة إسبانيا؟",
    "ما عاصمة روسيا؟",
    "ما هو أكبر كوكب في المجموعة الشمسية؟",
    "كم عدد قارات العالم؟",
    "ما هو لون الدم في جسم الإنسان؟",
    "ما هو الغاز الذي نتنفسه؟",
    "كيف تتكون الأمطار؟",
    "كم حاصل 5 زائد 3؟",
    "كم حاصل 20 ناقص 7؟",
    "كم حاصل 4 في 6؟",
    "كم حاصل 100 مقسوم على 4؟",
    "ما هو ناتج 9 ضعف 3؟",
    "عرف كلمة 'الشجاعة' في جملة.",
    "عرف كلمة 'الصبر' في جملة.",
    "عرف كلمة 'العلم' في جملة.",
    "عرف كلمة 'الكتاب' في جملة.",
    "عرف كلمة 'الصداقة' في جملة.",
    "اكتب قصة قصيرة عن رحلة بحرية.",
    "اكتب قصة قصيرة عن طفل ضاع في الغابة.",
    "اكتب قصة قصيرة عن مدينة المستقبل.",
    "اكتب قصة قصيرة عن رجل وقطته.",
    "اكتب قصة قصيرة عن عائلة في رمضان.",
    "اكتب قصيدة قصيرة عن البحر.",
    "اكتب قصيدة قصيرة عن الوطن.",
    "اكتب قصيدة قصيرة عن الخريف.",
    "ترجم إلى الإنجليزية: أهلاً وسهلاً.",
    "ترجم إلى الإنجليزية: أحب تعلم اللغات.",
    "ترجم إلى الإنجليزية: الشمس مشرقة اليوم.",
    "ترجم إلى الإنجليزية: القراءة مفيدة.",
    "ترجم إلى الإنجليزية: أنا ذاهب إلى المدرسة.",
    "ترجم إلى العربية: The book is on the table.",
    "ترجم إلى العربية: She is a doctor.",
    "ترجم إلى العربية: We love our family.",
    "لماذا الماء مهم للحياة؟",
    "ما هي فوائد الرياضة؟",
    "كيف تحافظ على الصحة؟",
    "ما هي أهمية التعليم؟",
    "لماذا نحتاج إلى الأصدقاء؟",
    "اذكر ثلاثة أنواع من الفواكه.",
    "اكتب جملة تحتوي على كلمة 'القمر'.",
    "اذكر عاصمتين عربيتين.",
    "لخص أهمية الشمس في جملة واحدة.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="weights_hf")
    ap.add_argument("--base_model", default="aubmindlab/aragpt2-large")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
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

    prompts = PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [json.loads(l)["instruction"] if isinstance(json.loads(l), dict) else json.loads(l)
                       for l in f if l.strip()]

    for q in prompts:
        qc = prep.preprocess(q) if prep else q
        text = f"### Instruction:\n{qc}\n\n### Response:\n"
        ids = tok.encode(text, return_tensors="pt").to(device)
        with th.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new,
                do_sample=True,
                top_k=args.top_k,
                temperature=args.temperature,
                repetition_penalty=1.3,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        if prep is not None and hasattr(prep, "desegment"):
            try:
                ans = prep.desegment(ans)
            except Exception:
                pass
        print(f"Q: {q}\n-> {ans}\n")

if __name__ == "__main__":
    main()
