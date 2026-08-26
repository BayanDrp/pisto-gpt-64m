import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
from generate import chat

QUESTIONS = [
    "ما عاصمة المملكة العربية السعودية؟",
    "ما عاصمة مصر؟",
    "ما عاصمة فرنسا؟",
    "كم عدد كواكب المجموعة الشمسية؟",
    "من هو أول رائد فضاء وطأت قدمه القمر؟",
    "ما هو الفرق بين النهار والليل؟",
    "اكتب جملة تحتوي على كلمة 'النجاح'.",
    "عرف كلمة 'الصداقة' في جملة قصيرة.",
    "اقترح ثلاثة أفكار لوجبة عشاء صحية.",
    "ترجم إلى الإنجليزية: الطقس جميل اليوم.",
    "كم حاصل 12 ناقص 4؟",
    "اكتب قصيدة قصيرة عن الربيع.",
    "ما هي لغة القرآن الكريم؟",
    "لخص بجملة واحدة أهمية قراءة الكتب.",
    "ما هو لون السماء في النهار؟",
    "اكتب قصة قصيرة عن صديقين وكلب في القرية.",
    "ما هو أصغر كواكب المجموعة الشمسية؟",
    "عرف كلمة 'العلم' في جملة واحدة.",
]


def is_degenerate(a: str) -> bool:
    """Heuristic first pass: reject empty / tiny / looped answers."""
    if not a or len(a.strip()) < 5:
        return True
    toks = a.split()
    if len(toks) > 8 and len(set(toks)) / len(toks) < 0.35:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="path to instruct_best.pt")
    ap.add_argument("--raw", default="data/sft_test_raw.jsonl")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "w", encoding="utf-8") as rf:
        for q in QUESTIONS:
            try:
                a = chat(q, weight_path=args.weight)
            except Exception as e:
                a = f"[ERROR {e}]"
            degen = is_degenerate(a)
            print("=" * 50)
            print("Q:", q)
            print("A:", a)
            print("FLAG:", "degenerate" if degen else "ok")
            rf.write(json.dumps({"instruction": q, "output": a,
                                  "degenerate": degen}, ensure_ascii=False) + "\n")

    print(f"\nRaw results -> {raw_path} ({len(QUESTIONS)} questions)")


if __name__ == "__main__":
    main()
