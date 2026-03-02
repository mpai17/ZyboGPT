"""ASCII character-level tokenizer for ZyboGPT.

Maps characters to token IDs 0-127 (standard ASCII).
Non-ASCII characters are mapped to '?' (63).
"""


class ASCIITokenizer:
    def __init__(self):
        self.vocab_size = 128
        self.pad_token = 0  # NUL

    def encode(self, text: str) -> list[int]:
        """Encode text to list of ASCII token IDs."""
        return [ord(c) if ord(c) < 128 else 63 for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return "".join(chr(t) if 0 <= t < 128 else "?" for t in tokens)

    def __len__(self) -> int:
        return self.vocab_size
