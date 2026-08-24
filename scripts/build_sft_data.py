import json
import os
import re
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Arabic helpers ───────────────────────────────────────────
AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def arabic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if AR.match(c))
    return ar / len(letters)


def norm(s: str) -> str:
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip().lower()


# ── Field extraction from heterogeneous HF schemas ──────────
INSTR_KEYS = ["instruction", "question", "prompt", "text"]
OUT_KEYS = ["output", "response", "answer", "completion"]
INPUT_KEYS = ["input", "context", "inputs"]


def _get(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return str(d[k])
    return None


def _parse_chat(instr_field, out_field):
    instr = None
    out = None
    for m in instr_field:
        role = str(m.get("role") or m.get("from") or "").lower()
        content = m.get("content") if "content" in m else m.get("value")
        if content in (None, ""):
            continue
        if role in ("user", "human") and instr is None:
            instr = str(content)
        elif role in ("assistant", "gpt", "bot") and out is None:
            out = str(content)
    if instr and out:
        return instr.strip(), "", out.strip()
    return None


def extract(row):
    # chat-format schemas
    if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
        res = _parse_chat(row["messages"], row["messages"])
        if res:
            return res
    if "conversations" in row:
        conv = row["conversations"]
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except Exception:
                conv = None
        if isinstance(conv, list) and len(conv) >= 2:
            res = _parse_chat(conv, conv)
            if res:
                return res
    instr = _get(row, INSTR_KEYS)
    out = _get(row, OUT_KEYS)
    inp = _get(row, INPUT_KEYS) or ""
    if not instr or not out:
        return None
    return instr.strip(), inp.strip(), out.strip()


def infer_task(instr: str, out: str) -> str:
    t = instr.lower()
    if any(w in t for w in ["ترجم", "translate", "إلى الإنجليزية", "بالإنجليزية", "بالانجليزية", "الإنجليزية"]):
        return "translate"
    if any(w in t for w in ["اضرب", "اقسم", "اجمع", "اطرح", "حاصل", "ضرب", "قسمة", "زائد", "ناقص", "جمع", "طرح"]):
        return "math"
    if any(w in t for w in ["عرّف", "عرف", "ما هو", "ما هي", "تعريف", "اشرح", "صف"]):
        return "define"
    if any(w in t for w in ["لخص", "ملخص"]):
        return "summarize"
    if any(w in t for w in ["صنف", "تصنيف"]):
        return "classify"
    return "mixed"


# ── Loaders ────────────────────────────────────────────────
def load_source(cfg_src, cfg):
    name = cfg_src["name"]
    limit = cfg_src.get("limit", 0)
    if name == "manual":
        path = REPO_ROOT / "data" / "manual_data.json"
        if not path.exists():
            print("  ! manual_data.json missing")
            return []
        data = json.load(open(path))
        items = []
        for pair in data:
            try:
                instr, out = pair[0], pair[1]
            except Exception:
                continue
            items.append((str(instr).strip(), "", str(out).strip()))
        print(f"  manual: {len(items)} pairs")
        return items
    try:
        from datasets import load_dataset
        if cfg_src.get("streaming"):
            ds = load_dataset(name, split="train", streaming=True)
            rows = list(ds.take(limit or 100000))
        else:
            ds = load_dataset(name, split="train")
            rows = list(ds)
        if limit:
            rows = rows[:limit]
        out = []
        for r in rows:
            e = extract(r)
            if e:
                out.append(e)
        print(f"  {name}: {len(out)} usable / {len(rows)} raw")
        return out
    except Exception as e:
        print(f"  ! failed to load {name}: {e}")
        return []


# ── Synthetic data ─────────────────────────────────────────
def gen_arithmetic(n):
    out = []
    ops = [
        ("زائد", lambda a, b: (a + b, f"{a} + {b}")),
        ("ناقص", lambda a, b: (a - b, f"{a} - {b}")),
        ("في", lambda a, b: (a * b, f"{a} * {b}")),
        ("على", lambda a, b: (a // b if b else 0, f"{a} // {b}")),
    ]
    tmpl = [
        "كم حاصل {a} {w} {b}؟",
        "احسب نتيجة {a} {w} {b}.",
        "ما قيمة {a} {w} {b}؟",
    ]
    for _ in range(n):
        a = random.randint(1, 200)
        b = random.randint(1, 50) if random.random() < 0.8 else random.randint(200, 999)
        if random.random() < 0.25:  # two-step
            c = random.randint(1, 50)
            a2 = random.randint(1, 100)
            ans = (a + b) - c
            out.append((f"إذا كان لديك {a} تفاحة ثم أضفت إليها {b} ثم أخذت {c}، كم تفاحة بقي لديك؟",
                        "", str(ans)))
            continue
        w, f = random.choice(ops)
        if w == "على" and b == 0:
            b = 1
        if w == "على" and a % b != 0:
            a = a - (a % b)
        ans = f(a, b)[0]
        instr = random.choice(tmpl).format(a=a, w=w, b=b)
        resp = str(ans) if random.random() < 0.6 else f"النتيجة هي {ans}."
        out.append((instr, "", resp))
    return out


def gen_factual(n):
    capitals = {
        "مصر": "القاهرة", "فرنسا": "باريس", "اليابان": "طوكيو", "السعودية": "الرياض",
        "الأردن": "عمّان", "سوريا": "دمشق", "العراق": "بغداد", "المغرب": "الرباط",
        "الجزائر": "الجزائر", "تونس": "تونس", "الكويت": "الكويت", "الإمارات": "أبوظبي",
        "تركيا": "أنقرة", "إيطاليا": "روما", "ألمانيا": "برلين", "إسبانيا": "مدريد",
        "روسيا": "موسكو", "الصين": "بكين", "الهند": "نيودلهي", "البرازيل": "برازيليا",
        "كندا": "أوتاوا", "المكسيك": "مدينة مكسيكو", "السويد": "ستوكهولم", "سويسرا": "برن",
        "إيران": "طهران", "باكستان": "إسلام أباد", "إندونيسيا": "جاكرتا", "كوريا الجنوبية": "سيول",
        "جنوب إفريقيا": "بريتوريا", "النرويج": "أوسلو", "اليونان": "أثينا", "البرتغال": "لشبونة",
    }
    facts = [
        ("كم عدد كواكب المجموعة الشمسية؟", "ثمانية كواكب."),
        ("ما هو أكبر كوكب في المجموعة الشمسية؟", "كوكب المشتري."),
        ("ما هو لون السماء في النهار؟", "الأزرق."),
        ("كم عدد قارات العالم؟", "سبع قارات."),
        ("ما هي عاصمة دولة تقع على نهر النيل؟", "القاهرة."),
        ("كم عدد أضلاع المثلث؟", "ثلاثة أضلاع."),
        ("ما هو العكس من الكلمة الساخن؟", "البارد."),
        ("كم يومًا في السنة الميلادية؟", "365 يومًا."),
        ("ما هي لغة القرآن الكريم؟", "اللغة العربية."),
        ("كم عدد حروف اللغة العربية؟", "ثمانون حرفًا."),
    ]
    out = []
    for _ in range(n // 2):
        c, cap = random.choice(list(capitals.items()))
        out.append((f"ما عاصمة {c}؟", "", cap))
    while len(out) < n:
        q, a = random.choice(facts)
        out.append((q, "", a))
    return out[:n]


# ── Main ───────────────────────────────────────────────────
def main():
    cfg_name = os.getenv("SFT_CONFIG", "sft.json")
    cfg = json.load(open(REPO_ROOT / "config" / cfg_name))
    dcfg = cfg["dataset"]
    random.seed(42)

    print("Loading sources...")
    raw = []
    for src in dcfg["sources"]:
        raw += load_source(src, cfg)
    # synthetic
    syn = gen_arithmetic(dcfg["synthetic"]["arithmetic"]) + gen_factual(dcfg["synthetic"]["factual"])
    print(f"synthetic: {len(syn)} pairs")

    # normalize + filter raw
    items = []
    seen = set()
    for instr, inp, out in raw + syn:
        if arabic_ratio(instr) < dcfg["arabic_ratio"] and arabic_ratio(out) < dcfg["arabic_ratio"]:
            continue
        if not (dcfg["min_instr_len"] <= len(instr) <= dcfg["max_instr_len"]):
            continue
        if not (dcfg["min_out_len"] <= len(out) <= dcfg["max_out_len"]):
            continue
        if len(instr) + len(out) < dcfg["min_total_len"]:
            continue
        key = norm(instr) + "|" + norm(out)
        if key in seen:
            continue
        seen.add(key)
        task = infer_task(instr, out)
        items.append({"instruction": instr, "input": inp, "output": out, "task": task})
    print(f"filtered total: {len(items)} samples")

    # balance by task (upsample rare, downsample huge)
    by_task = defaultdict(list)
    for it in items:
        by_task[it["task"]].append(it)
    floor = dcfg["floor_per_task"]
    ceil = dcfg["ceil_per_task"]
    balanced = []
    for task, lst in by_task.items():
        if len(lst) < floor:
            lst = [random.choice(lst) for _ in range(floor)]
        elif len(lst) > ceil:
            lst = random.sample(lst, ceil)
        balanced += lst
        print(f"  task '{task}': {len(lst)}")
    random.shuffle(balanced)

    # stratified split
    train_set, eval_set = [], []
    for task, lst in by_task.items():
        random.shuffle(lst)
        n_eval = max(1, int(len(lst) * (1 - dcfg["train_split"])))
        eval_set += lst[:n_eval]
        train_set += lst[n_eval:]
    random.shuffle(train_set)
    random.shuffle(eval_set)

    train_path = REPO_ROOT / dcfg["train_file"]
    eval_path = REPO_ROOT / dcfg["eval_file"]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for it in train_set:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for it in eval_set:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"WROTE train={len(train_set)} -> {train_path}")
    print(f"WROTE eval ={len(eval_set)} -> {eval_path}")


if __name__ == "__main__":
    main()
