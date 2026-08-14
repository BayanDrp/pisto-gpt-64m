"""Train a BPE tokenizer (vocab=8192) on TinyStories → config/bpe_tokenizer.json"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
sys.path.insert(0, str(_PROJ / "llm"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    from datasets import load_dataset

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    for i, s in enumerate(ds):
        if i >= 50000:
            break
        t = s.get("text", "")
        if len(t) > 100:
            texts.append(t)

    print(f"Training BPE (vocab=8192) on {len(texts):,} stories...")
    tok = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=["<pad>", "<bos>", "<eos>", "<|endoftext|>"],
        show_progress=True,
    ))

    path = str(_PROJ / "config" / "bpe_tokenizer.json")
    tok.save(path)
    print(f"Saved → {path}")
    print(f"Vocab: {tok.get_vocab_size()}")

    # Verify
    from tokenizer import ByteTokenizer
    t = ByteTokenizer(path)
    ids = t.tokenize("Hello! This is a test.")
    back = t.detokenize(ids)
    print(f"Test: {back} ({len(ids)} tokens, vocab={t.vocab_size})")


if __name__ == "__main__":
    main()
