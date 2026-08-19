"""Train a BPE tokenizer (vocab=8192) on Arabic text -> config/bpe_tokenizer.json"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
sys.path.insert(0, str(_PROJ / "llm"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    from datasets import load_dataset

    print("Loading Arabic text (Jr23xd23/ArabicText-Large)...")
    ds = load_dataset("Jr23xd23/ArabicText-Large", split="train", streaming=True)
    texts = []
    for i, s in enumerate(ds):
        if i >= 100_000:
            break
        t = s.get("text", "")
        if len(t) > 50:
            texts.append(t)

    print(f"Training BPE (vocab=8192) on {len(texts):,} Arabic docs...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
        min_frequency=2,
        show_progress=True,
    ))

    path = str(_PROJ / "config" / "bpe_tokenizer.json")
    tok.save(path)
    print(f"Saved -> {path}")
    print(f"Vocab: {tok.get_vocab_size()}")

    # Verify
    from tokenizer import ByteTokenizer
    t = ByteTokenizer(path)
    ids = t.tokenize("مرحبا، كيف حالك؟")
    back = t.detokenize(ids)
    print(f"Test: {back} ({len(ids)} tokens, vocab={t.vocab_size})")
    print(f"Special tokens: PAD={t.pad_id}, BOS={t.bos_id}, EOS={t.eos_id}")


if __name__ == "__main__":
    main()
