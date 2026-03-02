"""Shakespeare dataset loader for ZyboGPT training."""

import os
import urllib.request

import torch
from torch.utils.data import Dataset

from .tokenizer import ASCIITokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_shakespeare() -> str:
    """Download Tiny Shakespeare dataset if not present. Returns file path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(filepath):
        print(f"Downloading Tiny Shakespeare to {filepath}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
    return filepath


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset for causal LM training.

    Each sample is a (ctx_len+1,) tensor of token IDs.
    x = sample[:-1], y = sample[1:] for next-token prediction.
    """

    def __init__(self, ctx_len: int = 128, split: str = "train", train_frac: float = 0.9):
        filepath = download_shakespeare()
        with open(filepath, "r") as f:
            text = f.read()

        tokenizer = ASCIITokenizer()
        data = tokenizer.encode(text)
        self.data = torch.tensor(data, dtype=torch.long)

        # Train/val split
        split_idx = int(len(self.data) * train_frac)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        self.ctx_len = ctx_len

    def __len__(self) -> int:
        return len(self.data) - self.ctx_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.ctx_len + 1]
        x = chunk[:-1]  # (ctx_len,)
        y = chunk[1:]   # (ctx_len,)
        return x, y
