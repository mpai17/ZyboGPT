#!/usr/bin/env python3
"""Generate text from a trained ZyboGPT model (float PyTorch inference).

Usage:
    python scripts/generate.py
    python scripts/generate.py --prompt "To be" --tokens 100 --temperature 0.5
    python scripts/generate.py --checkpoint checkpoints/phase1/best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from python.train.model import ZyboGPT
from python.train.tokenizer import ASCIITokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from ZyboGPT model")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2/best.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ZyboGPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = ASCIITokenizer()
    tokens = torch.tensor(tok.encode(args.prompt), dtype=torch.long)

    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)

    print(tok.decode(out))


if __name__ == "__main__":
    main()
