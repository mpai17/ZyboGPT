"""Export trained model to FPGA-ready formats.

Outputs:
1. 1.6-bit packed ternary weights (TerEffic scheme: 5 trits per byte)
2. BRAM initialization files (.mem for Vivado, .bin for direct)
3. SpinalHDL constants file
4. Full-precision parameters (embeddings, norms) as INT16 fixed-point
"""

import os
import struct

import numpy as np
import torch

from .config import ZyboGPTConfig
from .model import ZyboGPT
from .bitlinear import ternary_quantize


def pack_ternary_5per_byte(trits: np.ndarray) -> np.ndarray:
    """Pack ternary values {-1,0,+1} as {0,1,2} into bytes, 5 trits per byte.

    Encoding: value+1 gives {0,1,2}. Pack as base-3: b0 + 3*b1 + 9*b2 + 27*b3 + 81*b4.
    Max value = 2+6+18+54+162 = 242 < 256, fits in a byte.
    """
    flat = trits.flatten()
    # Pad to multiple of 5
    pad_len = (5 - len(flat) % 5) % 5
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=flat.dtype)])

    # Map {-1,0,+1} -> {0,1,2}
    mapped = (flat + 1).astype(np.uint8)

    packed = []
    for i in range(0, len(mapped), 5):
        byte = (
            int(mapped[i])
            + 3 * int(mapped[i + 1])
            + 9 * int(mapped[i + 2])
            + 27 * int(mapped[i + 3])
            + 81 * int(mapped[i + 4])
        )
        packed.append(byte)

    return np.array(packed, dtype=np.uint8)


def quantize_to_int16(tensor: torch.Tensor, frac_bits: int = 10) -> np.ndarray:
    """Quantize full-precision tensor to INT16 fixed-point.

    Format: Q5.10 (5 integer bits, 10 fractional bits, 1 sign bit).
    """
    scale = 2**frac_bits
    quantized = (tensor.float().numpy() * scale).round().clip(-32768, 32767).astype(np.int16)
    return quantized


def export_model(
    checkpoint_path: str,
    output_dir: str = "export",
    config: ZyboGPTConfig | None = None,
):
    """Export trained model to FPGA formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config is None:
        config = ckpt.get("config", ZyboGPTConfig())
    model = ZyboGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    weight_data = {}  # name -> packed bytes
    meta = {}  # name -> {shape, scale, original_numel}

    # --- Export ternary weights ---
    ternary_layers = []
    for name, module in model.named_modules():
        if hasattr(module, "get_ternary_weights"):
            ternary_layers.append((name, module))

    all_ternary_packed = bytearray()
    raw_ternary = {}  # name -> raw ternary matrix for HW packing
    bram_offset = 0

    for name, module in ternary_layers:
        w_ternary, w_scale = module.get_ternary_weights()
        w_np = w_ternary.numpy().astype(np.int8)
        packed = pack_ternary_5per_byte(w_np)

        weight_data[name] = packed
        raw_ternary[name] = w_np
        meta[name] = {
            "shape": list(w_np.shape),
            "scale_fp32": float(w_scale.item()),
            "scale_int16": int(quantize_to_int16(w_scale.unsqueeze(0))[0]),
            "packed_bytes": len(packed),
            "bram_offset": bram_offset,
        }
        all_ternary_packed.extend(packed)
        bram_offset += len(packed)
        print(f"  {name}: shape={w_np.shape}, packed={len(packed)} bytes, offset={meta[name]['bram_offset']}")

    # --- Export full-precision params (embeddings, norms) as INT16 ---
    full_prec_data = {}

    # Token embedding
    tok_emb = quantize_to_int16(model.tok_emb.weight.data)
    full_prec_data["tok_emb"] = tok_emb
    print(f"  tok_emb: shape={tok_emb.shape}, bytes={tok_emb.nbytes}")

    # Positional embedding
    pos_emb = quantize_to_int16(model.pos_emb.weight.data)
    full_prec_data["pos_emb"] = pos_emb
    print(f"  pos_emb: shape={pos_emb.shape}, bytes={pos_emb.nbytes}")

    # RMSNorm weights (clamp to [-2.0, 2.0] to match hw_mode training STE clamp)
    gamma_clamp = 2.0
    for i, layer in enumerate(model.layers):
        for norm_name in ["attn_norm", "ff_norm"]:
            norm = getattr(layer, norm_name)
            w = quantize_to_int16(norm.weight.data.clamp(-gamma_clamp, gamma_clamp))
            key = f"layer{i}_{norm_name}"
            full_prec_data[key] = w
            print(f"  {key}: shape={w.shape}, bytes={w.nbytes}")

    final_norm = quantize_to_int16(model.final_norm.weight.data.clamp(-gamma_clamp, gamma_clamp))
    full_prec_data["final_norm"] = final_norm

    # --- Write binary weight file ---
    with open(os.path.join(output_dir, "weights_ternary.bin"), "wb") as f:
        f.write(bytes(all_ternary_packed))
    print(f"\nTernary weights: {len(all_ternary_packed)} bytes total")

    # Write in fixed order matching reference_inference.py's read order:
    # tok_emb, pos_emb, layer{i}_attn_norm, layer{i}_ff_norm, final_norm
    full_write_order = ["tok_emb", "pos_emb"]
    for i in range(config.n_layers):
        full_write_order += [f"layer{i}_attn_norm", f"layer{i}_ff_norm"]
    full_write_order.append("final_norm")

    with open(os.path.join(output_dir, "weights_full.bin"), "wb") as f:
        for key in full_write_order:
            f.write(full_prec_data[key].tobytes())

    # --- Write Vivado .mem file (hex, one byte per line for ternary) ---
    with open(os.path.join(output_dir, "weights_ternary.mem"), "w") as f:
        for b in all_ternary_packed:
            f.write(f"{b:02x}\n")

    # Write 32-bit wide .mem for BRAM with per-TDot-block packing.
    # Each TDot load reads 2048 trits (32 units x 64 elements).
    # Packed: ceil(2048/5) = 410 bytes, word-aligned to 412 bytes.
    # FFN down projection is reordered to TDot-load-friendly [32, 64] blocks.
    hw_packed = _generate_hw_packed(raw_ternary, config)
    with open(os.path.join(output_dir, "weights_ternary_32b.mem"), "w") as f:
        for i in range(0, len(hw_packed), 4):
            word = hw_packed[i] | (hw_packed[i + 1] << 8) | (hw_packed[i + 2] << 16) | (hw_packed[i + 3] << 24)
            f.write(f"{word:08x}\n")
    print(f"  HW packed weights: {len(hw_packed)} bytes ({len(hw_packed) // 4} 32-bit words)")

    # --- Write full-precision params as .mem (16-bit words) ---
    with open(os.path.join(output_dir, "weights_full_16b.mem"), "w") as f:
        for key in sorted(full_prec_data.keys()):
            for val in full_prec_data[key].flatten():
                f.write(f"{int(val) & 0xFFFF:04x}\n")

    # --- Write separate .mem files for embeddings ---
    for key in ["tok_emb", "pos_emb"]:
        with open(os.path.join(output_dir, f"{key}_16b.mem"), "w") as f:
            for val in full_prec_data[key].flatten():
                f.write(f"{int(val) & 0xFFFF:04x}\n")

    # --- Write SpinalHDL constants ---
    _write_spinalhdl_constants(output_dir, config, meta, hw_packed, full_prec_data)

    # --- Write metadata JSON ---
    import json
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"ternary_layers": meta, "config": config.__dict__}, f, indent=2)

    print(f"\nExport complete -> {output_dir}/")
    return meta


def _generate_hw_packed(raw_ternary: dict, config: ZyboGPTConfig) -> bytearray:
    """Generate per-TDot-block-packed weights for hardware BRAM.

    Each TDot load reads numTDots * tdotWidth = 2048 trits.
    Each block is padded to word-aligned size (412 bytes).
    FFN down projection is reordered from row-major [64, 256] to
    TDot-load-friendly [32, 64] blocks ordered by (batch, pass).
    """
    num_tdots = 32
    tdot_width = 64
    trits_per_block = num_tdots * tdot_width  # 2048
    data_bytes = (trits_per_block + 4) // 5  # ceil(2048/5) = 410
    bytes_per_block = ((data_bytes + 3) // 4) * 4  # word-align = 412

    hw_packed = bytearray()

    proj_suffixes = [
        "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
        "ff.up", "ff.down",
    ]

    for layer_idx in range(config.n_layers):
        for proj_suffix in proj_suffixes:
            name = f"layers.{layer_idx}.{proj_suffix}"
            w = raw_ternary[name]  # shape [d_out, d_in]
            d_out, d_in = w.shape

            if proj_suffix == "ff.down":
                # Reorder from row-major [64, 256] to TDot-load blocks [32, 64]
                # Block order: (batch=0,pass=0), (0,1), (0,2), (0,3), (1,0), ...
                n_batches = d_out // num_tdots  # 2
                n_passes = d_in // tdot_width   # 4
                for b in range(n_batches):
                    for p in range(n_passes):
                        block = w[b * num_tdots : (b + 1) * num_tdots,
                                  p * tdot_width : (p + 1) * tdot_width]
                        block_packed = pack_ternary_5per_byte(block.flatten())
                        pad = bytes_per_block - len(block_packed)
                        if pad > 0:
                            block_packed = np.concatenate(
                                [block_packed, np.zeros(pad, dtype=np.uint8)]
                            )
                        hw_packed.extend(block_packed)
            else:
                # Attention projections [64, 64] and FFN up [256, 64]:
                # Row-major layout matches TDot access (each row = tdot_width)
                n_batches = d_out // num_tdots
                for b in range(n_batches):
                    block = w[b * num_tdots : (b + 1) * num_tdots, :]
                    block_packed = pack_ternary_5per_byte(block.flatten())
                    pad = bytes_per_block - len(block_packed)
                    if pad > 0:
                        block_packed = np.concatenate(
                            [block_packed, np.zeros(pad, dtype=np.uint8)]
                        )
                    hw_packed.extend(block_packed)

    return hw_packed


def _write_spinalhdl_constants(output_dir, config, meta, ternary_packed, full_prec_data):
    """Generate SpinalHDL Scala file with weight BRAM init data."""
    lines = [
        "package zybogpt",
        "",
        "import spinal.core._",
        "",
        "object WeightInit {",
        f"  val VOCAB_SIZE = {config.vocab_size}",
        f"  val D_MODEL = {config.d_model}",
        f"  val N_HEADS = {config.n_heads}",
        f"  val N_LAYERS = {config.n_layers}",
        f"  val D_FF = {config.d_ff}",
        f"  val CTX_LEN = {config.ctx_len}",
        f"  val HEAD_DIM = {config.head_dim}",
        "",
        f"  val TERNARY_BYTES = {len(ternary_packed)}",
        "",
        "  // BRAM offsets for ternary weight layers",
    ]

    for name, info in sorted(meta.items()):
        clean_name = name.replace(".", "_").upper()
        lines.append(f"  val OFFSET_{clean_name} = {info['bram_offset']}")
        lines.append(f"  val SCALE_{clean_name} = {info['scale_int16']}  // Q5.10 fixed-point")

    # Large arrays are loaded from .mem files instead of compiled-in to avoid
    # JVM method size limit. Only keep references to file paths.
    lines.append("")
    lines.append("  // Large arrays loaded from .mem files at elaboration time")
    lines.append(f"  // Ternary weights: {len(ternary_packed)} packed bytes -> weights_ternary_32b.mem")
    lines.append(f"  // Token embedding: {config.vocab_size * config.d_model} INT16 -> tok_emb_16b.mem")
    lines.append(f"  // Pos embedding: {config.ctx_len * config.d_model} INT16 -> pos_emb_16b.mem")

    # Scale values array (ordered by layer, then projection)
    proj_order = [
        "layers.{}.attn.q_proj", "layers.{}.attn.k_proj",
        "layers.{}.attn.v_proj", "layers.{}.attn.o_proj",
        "layers.{}.ff.up", "layers.{}.ff.down",
    ]
    scale_values = []
    for layer_idx in range(config.n_layers):
        for proj_tmpl in proj_order:
            proj_name = proj_tmpl.format(layer_idx)
            scale_values.append(meta[proj_name]["scale_int16"])
    lines.append("")
    lines.append("  // Scale values per ternary layer, in order:")
    lines.append("  // [layer0_q, layer0_k, layer0_v, layer0_o, layer0_up, layer0_down, ...]")
    sv_str = ", ".join(str(v) for v in scale_values)
    lines.append(f"  val scaleValues: Array[Int] = Array({sv_str})")

    # RMSNorm gamma weights as flat array
    # Order: layer0_attn_norm, layer0_ff_norm, layer1_attn_norm, layer1_ff_norm, final_norm
    norm_gammas = []
    for layer_idx in range(config.n_layers):
        for norm_name in ["attn_norm", "ff_norm"]:
            key = f"layer{layer_idx}_{norm_name}"
            norm_gammas.extend(full_prec_data[key].flatten().tolist())
    norm_gammas.extend(full_prec_data["final_norm"].flatten().tolist())
    lines.append("")
    lines.append("  // RMSNorm gamma weights as INT16 Q5.10, flat:")
    lines.append("  // [layer0_attn_norm(dModel), layer0_ff_norm(dModel), ..., final_norm(dModel)]")
    lines.append(f"  val normGammas: Array[Int] = Array(  // {len(norm_gammas)} INT16 values")
    for i in range(0, len(norm_gammas), 16):
        chunk = norm_gammas[i : i + 16]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"    {vals},")
    lines.append("  )")

    # Keep individual norm arrays for backward compatibility
    for key in sorted(full_prec_data.keys()):
        if "norm" in key:
            data = full_prec_data[key].flatten()
            lines.append("")
            lines.append(f"  val {key}: Array[Int] = Array(  // {len(data)} INT16 values")
            for i in range(0, len(data), 16):
                chunk = data[i : i + 16]
                vals = ", ".join(str(int(v)) for v in chunk)
                lines.append(f"    {vals},")
            lines.append("  )")

    lines.append("}")
    lines.append("")

    with open(os.path.join(output_dir, "WeightInit.scala"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="export")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output)
