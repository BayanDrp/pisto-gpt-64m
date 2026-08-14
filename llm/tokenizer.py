from pathlib import Path


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def __init__(self, path=None):
        if path is None:
            root = Path(__file__).parent.parent
            candidates = [
                root / "weights" / "bpe_tokenizer.json",
                root / "config" / "bpe_tokenizer.json",
            ]
            path = str(next((p for p in candidates if p.exists()), candidates[-1]))
        from tokenizers import Tokenizer, decoders
        self._tok = Tokenizer.from_file(path)
        self._tok.decoder = decoders.ByteLevel()
        self.vocab_size = self._tok.get_vocab_size()

    def tokenize(self, text: str) -> list[int]:
        return [self.bos_id] + self._tok.encode(text).ids + [self.eos_id]

    def detokenize(self, token_ids: list[int]) -> str:
        ids = [t for t in token_ids if t not in (self.pad_id, self.bos_id, self.eos_id)]
        return self._tok.decode(ids)
