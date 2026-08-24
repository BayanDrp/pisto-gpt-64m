import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "llm"))
import torch as th
import generate
from tokenizer import ByteTokenizer

AR_LETTERS = {"أ": "أ", "ب": "ب", "ج": "ج", "د": "د"}
MAP = {"ا": "أ", "أ": "أ", "ب": "ب", "ج": "ج", "د": "د", "a": "أ", "b": "ب", "c": "ج", "d": "د"}


def greedy(mdl, prompt, inp="", max_new=24, device=None):
    tok = mdl.tokenizer
    if inp:
        text = f"### Instruction:\n{prompt}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    ids = tok.tokenize(text)
    if ids[-1] == tok.eos_id:
        ids = ids[:-1]
    input_ids = th.tensor([ids], dtype=th.long, device=device)
    generated = []
    with th.no_grad():
        for _ in range(max_new):
            logits = mdl(input_ids)[:, -1, :]
            nid = int(logits.argmax(-1).item())
            if nid == tok.eos_id:
                break
            generated.append(nid)
            input_ids = th.cat([input_ids, th.tensor([[nid]], device=device)], dim=1)
    return tok.detokenize(generated).strip()


def eval_heldout(mdl, path, device, steps=200):
    tok = mdl.tokenizer
    rows = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        print("held-out: no data")
        return
    loss_f = 0.0
    correct = 0
    total = 0
    n = 0
    for r in rows[:steps]:
        text = f"### Instruction:\n{r.get('instruction','')}\n\n### Response:\n" + r.get("output", "")
        ids = tok.tokenize(text)
        if ids[0] != tok.bos_id:
            ids = [tok.bos_id] + ids
        if ids[-1] != tok.eos_id:
            ids = ids + [tok.eos_id]
        if len(ids) < 2:
            continue
        x = th.tensor([ids[:-1]], dtype=th.long, device=device)
        y = th.tensor([ids[1:]], dtype=th.long, device=device)
        with th.no_grad():
            logits = mdl(x)
            loss = th.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0
            )
        loss_f += loss.item()
        pred = logits.argmax(-1)
        mask = y != 0
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
        n += 1
    print(f"held-out eval_loss={loss_f/max(n,1):.4f} token_acc={correct/max(total,1):.3f}")


def first_letter(s):
    for c in s:
        if c in MAP:
            return MAP[c]
    return None


def eval_mcq(mdl, device):
    try:
        from datasets import load_dataset
        ds = load_dataset("arbml/CIDAR-MCQ-100", split="train")
    except Exception as e:
        print(f"MCQ benchmark skipped: {e}")
        return
    correct = 0
    total = 0
    for row in ds:
        q = row.get("instruction") or row.get("question") or row.get("text") or ""
        opts = row.get("options") or row.get("choices") or []
        if isinstance(opts, str):
            opts = [o for o in opts.split("\n") if o.strip()]
        gold = str(row.get("answer") or row.get("label") or "").strip()
        if not q or not opts:
            continue
        body = "\n".join(f"{chr(0x0623+i)}. {o}" for i, o in enumerate(opts))
        prompt = f"السؤال: {q}\nالخيارات:\n{body}\nالإجابة الصحيحة هي:"
        ans = greedy(mdl, prompt, max_new=12, device=device)
        pred = first_letter(ans)
        gold_letter = first_letter(gold)
        if pred and gold_letter and pred == gold_letter:
            correct += 1
        total += 1
    if total:
        print(f"MCQ accuracy: {correct/total:.3f} ({correct}/{total})")


def main():
    weight = os.getenv("SFT_WEIGHTS") or str((REPO_ROOT / "weights" / "instruct_best.pt"))
    if not Path(weight).exists():
        print(f"weights not found: {weight}")
        return
    mdl = generate.load_model(weight)
    device = next(mdl.parameters()).device
    mdl.eval()

    eval_heldout(mdl, REPO_ROOT / "data" / "sft_eval.jsonl", device)
    eval_mcq(mdl, device)

    smoke = [
        "ما عاصمة فرنسا؟",
        "كم حاصل 12 زائد 8؟",
        "من هو مؤلف رواية ألف ليلة وليلة؟",
        "اكتب جملة بالعربية عن الطقس.",
    ]
    print("\n=== generation smoke tests ===")
    for q in smoke:
        print(f"Q: {q}\nA: {greedy(mdl, q, device=device)}\n")


if __name__ == "__main__":
    main()
