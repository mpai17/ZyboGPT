#!/usr/bin/env python3
"""Generate BRAM initialization files from trained model.

Converts PyTorch checkpoint -> .coe/.mem files for Vivado BRAM init.
Also generates SpinalHDL weight constants.

Usage:
    python scripts/generate_weights.py checkpoints/best.pt --output export/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.train.export import export_model


def generate_coe(mem_file: str, coe_file: str, radix: int = 16):
    """Convert .mem file to Vivado .coe format."""
    with open(mem_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(coe_file, "w") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        for i, line in enumerate(lines):
            sep = ";" if i == len(lines) - 1 else ","
            f.write(f"{line}{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate FPGA weight files")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint .pt file")
    parser.add_argument("--output", default="export", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Exporting weights from {args.checkpoint}...")
    meta = export_model(args.checkpoint, args.output)

    # Generate .coe files for Vivado
    mem_files = [
        ("weights_ternary_32b.mem", "weights_ternary_32b.coe"),
        ("weights_full_16b.mem", "weights_full_16b.coe"),
    ]

    for mem_name, coe_name in mem_files:
        mem_path = os.path.join(args.output, mem_name)
        coe_path = os.path.join(args.output, coe_name)
        if os.path.exists(mem_path):
            generate_coe(mem_path, coe_path)
            print(f"  Generated {coe_path}")

    # Copy SpinalHDL constants to hw source
    scala_src = os.path.join(args.output, "WeightInit.scala")
    scala_dst = os.path.join("hw", "src", "main", "scala", "zybogpt", "WeightInit.scala")
    if os.path.exists(scala_src):
        import shutil
        shutil.copy2(scala_src, scala_dst)
        print(f"  Copied WeightInit.scala -> {scala_dst}")

    print("\nWeight generation complete!")
    print(f"  Ternary weights: {meta.get('total_packed_bytes', 'N/A')} bytes")
    print(f"  Output dir: {args.output}/")


if __name__ == "__main__":
    main()
