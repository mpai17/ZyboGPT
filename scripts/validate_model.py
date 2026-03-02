#!/usr/bin/env python3
"""Validate model: compare PyTorch inference vs INT8 reference.

Ensures the quantized reference inference produces reasonable output
before running RTL simulation.

Usage:
    python scripts/validate_model.py checkpoints/best.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.train.config import ZyboGPTConfig
from python.train.model import ZyboGPT
from python.train.tokenizer import ASCIITokenizer
from python.train.export import export_model
from python.train.reference_inference import (
    load_weights,
    reference_inference,
    generate_test_vectors,
)


def validate(checkpoint_path: str, export_dir: str = "export"):
    tokenizer = ASCIITokenizer()

    # Step 1: Load PyTorch model
    print("=" * 60)
    print("Step 1: PyTorch FP32 inference")
    print("=" * 60)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", ZyboGPTConfig())
    model = ZyboGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = "ROMEO:"
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)

    with torch.no_grad():
        fp32_tokens = model.generate(prompt_tokens, max_new_tokens=32, temperature=0)

    fp32_text = tokenizer.decode(fp32_tokens)
    print(f"  Prompt: {prompt}")
    print(f"  FP32 output: {fp32_text}")

    # Step 2: Export model
    print("\n" + "=" * 60)
    print("Step 2: Export to FPGA format")
    print("=" * 60)

    if not os.path.exists(export_dir):
        export_model(checkpoint_path, export_dir)
    else:
        print(f"  Using existing export: {export_dir}/")

    # Step 3: INT8 reference inference
    print("\n" + "=" * 60)
    print("Step 3: INT8 reference inference")
    print("=" * 60)

    weights, ref_config = load_weights(export_dir)
    prompt_ids = tokenizer.encode(prompt)
    ref_tokens, debug = reference_inference(prompt_ids, weights, ref_config, max_new_tokens=32)

    ref_text = tokenizer.decode(ref_tokens)
    print(f"  INT8 output: {ref_text}")

    # Step 4: Compare
    print("\n" + "=" * 60)
    print("Step 4: Comparison")
    print("=" * 60)

    fp32_gen = fp32_tokens[len(prompt):]
    ref_gen = ref_tokens

    matches = sum(a == b for a, b in zip(fp32_gen, ref_gen))
    total = min(len(fp32_gen), len(ref_gen))

    print(f"  Token match: {matches}/{total} ({100*matches/max(total,1):.1f}%)")
    print(f"  FP32 top-5 first token: {debug[0].get('logits_top5', 'N/A') if debug else 'N/A'}")

    if matches < total * 0.5:
        print("\n  WARNING: Low match rate. Quantization error may be significant.")
        print("  Consider adjusting quantization parameters or retraining.")
    else:
        print("\n  PASS: Quantized reference reasonably matches FP32.")

    # Step 5: Generate test vectors
    print("\n" + "=" * 60)
    print("Step 5: Generate RTL test vectors")
    print("=" * 60)

    test_dir = os.path.join(export_dir, "test_vectors")
    generated, _ = generate_test_vectors(
        export_dir, test_dir, num_tokens=122, temperature=0.5, seed=0xDEADBEEF
    )
    print(f"  Test vectors saved to {test_dir}/")

    return matches, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--export", default="export", help="Export directory")
    args = parser.parse_args()

    validate(args.checkpoint, args.export)
