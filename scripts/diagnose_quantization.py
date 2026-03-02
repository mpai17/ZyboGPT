#!/usr/bin/env python3
"""Diagnose per-stage information loss between float/hw_mode/INT8 forward passes.

Loads Phase 1 (float) and Phase 2 (hw_mode) checkpoints, runs a forward pass
on "ROMEO:" position 5 (':'), and compares activations at every stage:
  - Float model (full precision)
  - HW-mode model (quantized PyTorch)
  - INT8 reference inference (bit-accurate integer arithmetic)

Reports dynamic range, entropy, cosine similarity, unique values, and
histogram concentration per stage.
"""

import json
import math
import os
import sys

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from python.train.config import ZyboGPTConfig
from python.train.model import ZyboGPT, RMSNorm, Attention
from python.train.bitlinear import (
    ternary_quantize, ste_round_clamp_int8, ste_shift_right,
    ste_round_clamp, ste_hw_int8, set_hw_truncation,
)
from python.train.reference_inference import (
    load_weights, hw_rmsnorm, hw_ternary_matvec, hw_ternary_matvec_accum,
    hw_softmax, to_sint, clamp_int8, INV_SQRT_LUT,
)


# ================================================================
# Statistics helpers
# ================================================================

def compute_entropy(x):
    """Compute Shannon entropy (bits) of a discrete distribution of values."""
    vals, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64).flatten()
    b = np.array(b, dtype=np.float64).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def histogram_summary(x, name=""):
    """Return a compact histogram description for INT8-range values."""
    x = np.array(x, dtype=np.float64).flatten()
    bins = [(-128, -64), (-64, -32), (-32, -16), (-16, -8), (-8, -4),
            (-4, 0), (0, 0), (0, 4), (4, 8), (8, 16), (16, 32),
            (32, 64), (64, 128)]
    counts = {}
    for lo, hi in bins:
        if lo == hi == 0:
            c = int(np.sum(x == 0))
            if c > 0:
                counts["=0"] = c
        else:
            c = int(np.sum((x >= lo) & (x < hi)))
            if c > 0:
                counts[f"[{lo},{hi})"] = c
    return counts


def report_stage(name, float_act, hw_act, int8_act):
    """Print comprehensive per-stage comparison statistics."""
    float_np = np.array(float_act, dtype=np.float64).flatten()
    hw_np = np.array(hw_act, dtype=np.float64).flatten()
    int8_np = np.array(int8_act, dtype=np.float64).flatten()

    n = len(float_np)

    print(f"\n{'='*72}")
    print(f"  STAGE: {name}  (dim={n})")
    print(f"{'='*72}")

    # Float statistics
    print(f"  FLOAT  : min={float_np.min():10.4f}  max={float_np.max():10.4f}"
          f"  std={float_np.std():8.4f}  mean={float_np.mean():8.4f}")

    # HW-mode (PyTorch quantized) statistics
    print(f"  HW_MODE: min={hw_np.min():10.1f}  max={hw_np.max():10.1f}"
          f"  std={hw_np.std():8.2f}  mean={hw_np.mean():8.2f}"
          f"  unique={len(np.unique(hw_np)):4d}/{n}")

    # INT8 reference statistics
    print(f"  INT8   : min={int8_np.min():10.1f}  max={int8_np.max():10.1f}"
          f"  std={int8_np.std():8.2f}  mean={int8_np.mean():8.2f}"
          f"  unique={len(np.unique(int8_np)):4d}/{n}")

    # Entropy
    hw_entropy = compute_entropy(hw_np.astype(int))
    int8_entropy = compute_entropy(int8_np.astype(int))
    max_entropy = math.log2(max(n, 2))
    print(f"  ENTROPY: hw_mode={hw_entropy:.2f} bits  int8={int8_entropy:.2f} bits"
          f"  (max possible={max_entropy:.2f} bits for {n} elements)")

    # Cosine similarity
    cos_float_hw = cosine_sim(float_np, hw_np)
    cos_float_int8 = cosine_sim(float_np, int8_np)
    cos_hw_int8 = cosine_sim(hw_np, int8_np)
    print(f"  COS_SIM: float<->hw={cos_float_hw:.6f}"
          f"  float<->int8={cos_float_int8:.6f}"
          f"  hw<->int8={cos_hw_int8:.6f}")

    # Relative error (float -> hw)
    if np.abs(float_np).max() > 1e-8:
        # Scale float to same range as hw for fair comparison
        scale = np.abs(hw_np).max() / max(np.abs(float_np).max(), 1e-8)
        float_scaled = float_np * scale
        mse = np.mean((float_scaled - hw_np) ** 2)
        rmse = math.sqrt(mse)
        print(f"  RMSE(float_scaled->hw)={rmse:.4f}  (scale_factor={scale:.4f})")

    # INT8 utilization: what fraction of [-128,127] range is used?
    hw_range = hw_np.max() - hw_np.min()
    int8_range = int8_np.max() - int8_np.min()
    print(f"  RANGE_UTIL: hw_mode={hw_range:.0f}/255={hw_range/255*100:.1f}%"
          f"  int8={int8_range:.0f}/255={int8_range/255*100:.1f}%")

    # Saturation count (values hitting -128 or 127)
    hw_sat = int(np.sum(np.abs(hw_np) >= 127))
    int8_sat = int(np.sum(np.abs(int8_np) >= 127))
    print(f"  SATURATED: hw_mode={hw_sat}/{n} ({hw_sat/n*100:.1f}%)"
          f"  int8={int8_sat}/{n} ({int8_sat/n*100:.1f}%)")

    # Zero count
    hw_zeros = int(np.sum(hw_np == 0))
    int8_zeros = int(np.sum(int8_np == 0))
    print(f"  ZEROS  : hw_mode={hw_zeros}/{n} ({hw_zeros/n*100:.1f}%)"
          f"  int8={int8_zeros}/{n} ({int8_zeros/n*100:.1f}%)")

    # Compact histogram for hw_mode
    hist = histogram_summary(hw_np)
    print(f"  HIST(hw): {hist}")

    return {
        "name": name,
        "cos_float_hw": cos_float_hw,
        "cos_float_int8": cos_float_int8,
        "cos_hw_int8": cos_hw_int8,
        "hw_entropy": hw_entropy,
        "int8_entropy": int8_entropy,
        "hw_range_util": hw_range / 255,
        "int8_range_util": int8_range / 255,
        "hw_saturated_pct": hw_sat / n * 100,
        "int8_saturated_pct": int8_sat / n * 100,
        "float_std": float(float_np.std()),
        "hw_std": float(hw_np.std()),
        "int8_std": float(int8_np.std()),
        "hw_unique": int(len(np.unique(hw_np))),
        "int8_unique": int(len(np.unique(int8_np))),
    }


# ================================================================
# Float model forward pass with activation capture
# ================================================================

@torch.no_grad()
def float_forward_with_hooks(model, token_id, position, config):
    """Run float model forward pass, capturing intermediate activations."""
    acts = {}
    device = next(model.parameters()).device

    tokens = torch.tensor([[token_id]], device=device)
    positions = torch.tensor([position], device=device)

    # Embedding
    tok_raw = model.tok_emb(tokens)  # (1, 1, d_model)
    pos_raw = model.pos_emb(positions)  # (1, d_model)
    x = tok_raw + pos_raw  # (1, 1, d_model)
    acts["embedding"] = x.squeeze().cpu().numpy()

    # Layer loop
    for layer_idx, layer in enumerate(model.layers):
        # Attention norm
        h = layer.attn_norm(x)
        acts[f"L{layer_idx}_attn_norm"] = h.squeeze().cpu().numpy()

        # QKV projections
        B, T, C = h.shape
        q = layer.attn.q_proj(h)
        acts[f"L{layer_idx}_q_proj"] = q.squeeze().cpu().numpy()
        k = layer.attn.k_proj(h)
        acts[f"L{layer_idx}_k_proj"] = k.squeeze().cpu().numpy()
        v = layer.attn.v_proj(h)
        acts[f"L{layer_idx}_v_proj"] = v.squeeze().cpu().numpy()

        # For single-token at position 0-5 (prefill): use full attention
        # Run full attention
        attn_out, new_kv = layer.attn(h, mask=None, kv_cache=None)
        acts[f"L{layer_idx}_attn_out"] = attn_out.squeeze().cpu().numpy()

        # Residual
        x = x + attn_out
        acts[f"L{layer_idx}_attn_residual"] = x.squeeze().cpu().numpy()

        # FF norm
        h = layer.ff_norm(x)
        acts[f"L{layer_idx}_ff_norm"] = h.squeeze().cpu().numpy()

        # FFN
        up = layer.ff.up(h)
        acts[f"L{layer_idx}_ff_up_raw"] = up.squeeze().cpu().numpy()
        up_relu = F.relu(up)
        acts[f"L{layer_idx}_ff_up_relu"] = up_relu.squeeze().cpu().numpy()
        down = layer.ff.down(up_relu)
        acts[f"L{layer_idx}_ff_down"] = down.squeeze().cpu().numpy()

        # Residual
        x = x + down
        acts[f"L{layer_idx}_ff_residual"] = x.squeeze().cpu().numpy()

    # Final norm
    x_norm = model.final_norm(x)
    acts["final_norm"] = x_norm.squeeze().cpu().numpy()

    # Logits
    logits = model.output_head(x_norm) * (config.d_model ** -0.5)
    acts["logits"] = logits.squeeze().cpu().numpy()

    return acts


# ================================================================
# HW-mode model forward pass with activation capture
# ================================================================

@torch.no_grad()
def hw_forward_with_hooks(model, token_id, position, config):
    """Run hw_mode model forward pass, capturing intermediate activations."""
    acts = {}
    device = next(model.parameters()).device

    tokens = torch.tensor([[token_id]], device=device)
    positions = torch.tensor([position], device=device)

    # Embedding (hw mode)
    tok_raw = model.tok_emb(tokens)
    pos_raw = model.pos_emb(positions)
    tok_q = ste_round_clamp(tok_raw * 1024, -32768, 32767)
    pos_q = ste_round_clamp(pos_raw * 1024, -32768, 32767)
    x = ste_shift_right(tok_q + pos_q, 3)
    x = ste_hw_int8(x)
    acts["embedding"] = x.squeeze().cpu().numpy()

    # Layer loop
    for layer_idx, layer in enumerate(model.layers):
        # Attention norm
        h = layer.attn_norm(x)
        acts[f"L{layer_idx}_attn_norm"] = h.squeeze().cpu().numpy()

        # QKV projections
        q = layer.attn.q_proj(h)
        acts[f"L{layer_idx}_q_proj"] = q.squeeze().cpu().numpy()
        k = layer.attn.k_proj(h)
        acts[f"L{layer_idx}_k_proj"] = k.squeeze().cpu().numpy()
        v = layer.attn.v_proj(h)
        acts[f"L{layer_idx}_v_proj"] = v.squeeze().cpu().numpy()

        # Full attention
        attn_out, new_kv = layer.attn(h, mask=None, kv_cache=None)
        acts[f"L{layer_idx}_attn_out"] = attn_out.squeeze().cpu().numpy()

        # Residual
        x = x + attn_out
        x = ste_hw_int8(x)
        acts[f"L{layer_idx}_attn_residual"] = x.squeeze().cpu().numpy()

        # FF norm
        h = layer.ff_norm(x)
        acts[f"L{layer_idx}_ff_norm"] = h.squeeze().cpu().numpy()

        # FFN
        up = layer.ff.up(h)
        acts[f"L{layer_idx}_ff_up_raw"] = up.squeeze().cpu().numpy()
        up_relu = F.relu(up)
        up_relu = ste_round_clamp_int8(up_relu)
        acts[f"L{layer_idx}_ff_up_relu"] = up_relu.squeeze().cpu().numpy()
        down = layer.ff.down(up_relu)
        acts[f"L{layer_idx}_ff_down"] = down.squeeze().cpu().numpy()

        # Residual
        x = x + down
        x = ste_hw_int8(x)
        acts[f"L{layer_idx}_ff_residual"] = x.squeeze().cpu().numpy()

    # Final norm
    x_norm = model.final_norm(x)
    acts["final_norm"] = x_norm.squeeze().cpu().numpy()

    # Logits (hw)
    emb_q = ste_round_clamp(model.tok_emb.weight * 1024, -32768, 32767)
    logits_raw = torch.matmul(x_norm, emb_q.T)
    logits = logits_raw / (64.0 * 1024.0 * math.sqrt(config.d_model))
    acts["logits"] = logits.squeeze().cpu().numpy()
    acts["logits_raw_int24"] = logits_raw.squeeze().cpu().numpy()

    return acts


# ================================================================
# INT8 reference inference single-step with activation capture
# ================================================================

def int8_forward_single_step(token_id, position, weights, config):
    """Run bit-accurate INT8 reference for a single token, capturing activations."""
    acts = {}
    d = config.d_model
    n_h = config.n_heads
    h_d = config.head_dim

    # Embedding
    act = np.zeros(d, dtype=np.int8)
    for i in range(d):
        tok_val = int(weights["tok_emb"][token_id, i])
        pos_val = int(weights["pos_emb"][position, i])
        emb_sum = tok_val + pos_val
        act[i] = clamp_int8(emb_sum >> 3)
    acts["embedding"] = act.copy()

    # We need KV cache from all prompt tokens for proper attention
    # For position 5 (":" in "ROMEO:"), we need to process all prior tokens too
    prompt = [ord(c) for c in "ROMEO:"]
    kv_cache = [[[] for _ in range(n_h)] for _ in range(config.n_layers)]

    # Process all tokens up to and including position
    for step in range(position + 1):
        tok = prompt[step]
        pos = step

        # Embedding
        step_act = np.zeros(d, dtype=np.int8)
        for i in range(d):
            tok_val = int(weights["tok_emb"][tok, i])
            pos_val = int(weights["pos_emb"][pos, i])
            emb_sum = tok_val + pos_val
            step_act[i] = clamp_int8(emb_sum >> 3)

        if step == position:
            acts["embedding"] = step_act.copy()

        for layer_idx in range(config.n_layers):
            prefix = f"layers.{layer_idx}"

            # Attn norm
            gamma = weights[f"layer{layer_idx}_attn_norm"]
            normed = hw_rmsnorm(step_act, gamma)
            if step == position:
                acts[f"L{layer_idx}_attn_norm"] = normed.copy()

            # QKV
            q = hw_ternary_matvec(normed, weights[f"{prefix}.attn.q_proj"])
            k = hw_ternary_matvec(normed, weights[f"{prefix}.attn.k_proj"])
            v = hw_ternary_matvec(normed, weights[f"{prefix}.attn.v_proj"])
            if step == position:
                acts[f"L{layer_idx}_q_proj"] = q.copy()
                acts[f"L{layer_idx}_k_proj"] = k.copy()
                acts[f"L{layer_idx}_v_proj"] = v.copy()

            # Store KV
            for h in range(n_h):
                k_head = k[h * h_d:(h + 1) * h_d].copy()
                v_head = v[h * h_d:(h + 1) * h_d].copy()
                if pos < len(kv_cache[layer_idx][h]):
                    kv_cache[layer_idx][h][pos] = (k_head, v_head)
                else:
                    kv_cache[layer_idx][h].append((k_head, v_head))

            # Attention per head
            attn_out = np.zeros(d, dtype=np.int8)
            all_scores = {}
            all_probs = {}
            for h in range(n_h):
                scores = [0] * config.ctx_len
                for p in range(pos + 1):
                    k_cached = kv_cache[layer_idx][h][p][0]
                    dot = 0
                    for i in range(h_d):
                        dot += int(q[h * h_d + i]) * int(k_cached[i])
                    dot = to_sint(dot, 24)
                    scores[p] = to_sint((dot * 45) >> 8, 16)
                for i in range(pos + 1, config.ctx_len):
                    scores[i] = -32767

                probs = hw_softmax(scores, pos + 1, config.ctx_len)

                if step == position:
                    all_scores[h] = scores[:pos + 1]
                    all_probs[h] = probs[:pos + 1]

                acc = [0] * h_d
                for p in range(pos + 1):
                    v_cached = kv_cache[layer_idx][h][p][1]
                    prob_val = probs[p]
                    for dd in range(h_d):
                        v_val = to_sint(int(v_cached[dd]), 16)
                        product = to_sint(prob_val * v_val, 24)
                        acc[dd] = to_sint(acc[dd] + product, 24)
                for dd in range(h_d):
                    attn_out[h * h_d + dd] = clamp_int8(acc[dd] >> 8)

            if step == position:
                acts[f"L{layer_idx}_attn_scores"] = all_scores
                acts[f"L{layer_idx}_attn_probs"] = all_probs
                acts[f"L{layer_idx}_attn_out_pre_oproj"] = attn_out.copy()

            # O proj
            o_out = hw_ternary_matvec(attn_out, weights[f"{prefix}.attn.o_proj"])
            if step == position:
                acts[f"L{layer_idx}_attn_out"] = o_out.copy()

            # Residual
            for i in range(d):
                s = to_sint(int(step_act[i]), 16) + to_sint(int(o_out[i]), 16)
                step_act[i] = clamp_int8(s)
            if step == position:
                acts[f"L{layer_idx}_attn_residual"] = step_act.copy()

            # FF norm
            gamma = weights[f"layer{layer_idx}_ff_norm"]
            normed = hw_rmsnorm(step_act, gamma)
            if step == position:
                acts[f"L{layer_idx}_ff_norm"] = normed.copy()

            # FFN up + relu
            up = hw_ternary_matvec(normed, weights[f"{prefix}.ff.up"])
            if step == position:
                acts[f"L{layer_idx}_ff_up_raw"] = up.copy()
            for i in range(len(up)):
                if up[i] < 0:
                    up[i] = 0
            if step == position:
                acts[f"L{layer_idx}_ff_up_relu"] = up.copy()

            # FFN down
            down = hw_ternary_matvec_accum(up, weights[f"{prefix}.ff.down"], d_in_slice=d)
            if step == position:
                acts[f"L{layer_idx}_ff_down"] = down.copy()

            # Residual
            for i in range(d):
                s = to_sint(int(step_act[i]), 16) + to_sint(int(down[i]), 16)
                step_act[i] = clamp_int8(s)
            if step == position:
                acts[f"L{layer_idx}_ff_residual"] = step_act.copy()

    # Final norm
    gamma = weights["final_norm"]
    normed_final = hw_rmsnorm(step_act, gamma)
    acts["final_norm"] = normed_final.copy()

    # Logits
    all_logits = []
    for v in range(config.vocab_size):
        logit_acc = 0
        for i in range(d):
            q_val = to_sint(int(normed_final[i]), 24)
            e_val = to_sint(int(weights["tok_emb"][v, i]), 24)
            product = q_val * e_val
            logit_acc = to_sint(logit_acc + product, 24)
        all_logits.append(logit_acc)
    acts["logits"] = np.array(all_logits, dtype=np.float64)
    acts["logits_raw_int24"] = np.array(all_logits, dtype=np.float64)

    return acts


# ================================================================
# Attention diagnostic for float model (single-token)
# ================================================================

@torch.no_grad()
def float_attention_detail(model, prompt_str, target_pos, config):
    """Get float attention scores and probs for target position."""
    device = next(model.parameters()).device
    prompt = [ord(c) for c in prompt_str]
    tokens = torch.tensor([prompt], device=device)

    # Run full model to get activations at the target position
    B, T = tokens.shape
    positions = torch.arange(0, T, device=device)

    if config.hw_mode:
        tok_raw = model.tok_emb(tokens)
        pos_raw = model.pos_emb(positions)
        tok_q = ste_round_clamp(tok_raw * 1024, -32768, 32767)
        pos_q = ste_round_clamp(pos_raw * 1024, -32768, 32767)
        x = ste_shift_right(tok_q + pos_q, 3)
        x = ste_hw_int8(x)
    else:
        x = model.tok_emb(tokens) + model.pos_emb(positions)

    details = {}
    for layer_idx, layer in enumerate(model.layers):
        h = layer.attn_norm(x)
        B, T, C = h.shape
        n_h = config.n_heads
        h_d = config.head_dim

        q = layer.attn.q_proj(h).view(B, T, n_h, h_d).transpose(1, 2)
        k = layer.attn.k_proj(h).view(B, T, n_h, h_d).transpose(1, 2)
        v = layer.attn.v_proj(h).view(B, T, n_h, h_d).transpose(1, 2)

        # Compute attention scores
        if config.hw_mode:
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = ste_shift_right(scores * 45, 8)
            # Causal mask
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, -32767)
            probs_uint8 = layer.attn._hw_softmax(scores)
            details[f"L{layer_idx}_scores"] = scores[0, :, target_pos, :target_pos+1].cpu().numpy()
            details[f"L{layer_idx}_probs"] = probs_uint8[0, :, target_pos, :target_pos+1].cpu().numpy()
        else:
            scale = 1.0 / math.sqrt(h_d)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float('-inf'))
            probs = F.softmax(scores, dim=-1)
            details[f"L{layer_idx}_scores"] = scores[0, :, target_pos, :target_pos+1].cpu().numpy()
            details[f"L{layer_idx}_probs"] = probs[0, :, target_pos, :target_pos+1].cpu().numpy()

        attn_out, _ = layer.attn(h, mask=True, kv_cache=None)
        x = x + attn_out
        if config.hw_mode:
            x = ste_hw_int8(x)
        h = layer.ff_norm(x)
        x = x + layer.ff(h)
        if config.hw_mode:
            x = ste_hw_int8(x)

    return details


# ================================================================
# Main diagnostic
# ================================================================

def main():
    CHECKPOINT_PHASE1 = os.path.join(PROJECT_ROOT, "checkpoints", "phase1", "best.pt")
    CHECKPOINT_PHASE2 = os.path.join(PROJECT_ROOT, "checkpoints", "phase2", "best.pt")
    EXPORT_DIR = os.path.join(PROJECT_ROOT, "export")

    prompt_str = "ROMEO:"
    target_pos = 5  # the ':' token
    target_token = ord(':')

    print("=" * 72)
    print("  ZyboGPT Quantization Diagnostic")
    print(f"  Prompt: '{prompt_str}', analyzing position {target_pos} (token='{chr(target_token)}'={target_token})")
    print("=" * 72)

    # ----------------------------------------------------------------
    # Load Phase 1 (float) model
    # ----------------------------------------------------------------
    print("\n[1/4] Loading Phase 1 (float) checkpoint...")
    cp1 = torch.load(CHECKPOINT_PHASE1, map_location="cpu", weights_only=False)
    config_float = cp1["config"]
    config_float.hw_mode = False  # Ensure float mode
    model_float = ZyboGPT(config_float)
    model_float.load_state_dict(cp1["model"])
    model_float.eval()
    print(f"  Loaded. Step={cp1['step']}, hw_mode={config_float.hw_mode}")

    # ----------------------------------------------------------------
    # Load Phase 2 (hw_mode) model
    # ----------------------------------------------------------------
    print("\n[2/4] Loading Phase 2 (hw_mode) checkpoint...")
    cp2 = torch.load(CHECKPOINT_PHASE2, map_location="cpu", weights_only=False)
    config_hw = cp2["config"]
    config_hw.hw_mode = True
    set_hw_truncation(False)  # Use clamping (Phase 2 trained with clamping)
    model_hw = ZyboGPT(config_hw)
    model_hw.load_state_dict(cp2["model"])
    model_hw.eval()
    print(f"  Loaded. Step={cp2['step']}, hw_mode={config_hw.hw_mode}")

    # ----------------------------------------------------------------
    # Load INT8 reference weights
    # ----------------------------------------------------------------
    print("\n[3/4] Loading INT8 reference weights from export...")
    int8_weights, int8_config = load_weights(EXPORT_DIR)
    print(f"  Loaded. vocab={int8_config.vocab_size}, d_model={int8_config.d_model}")

    # ----------------------------------------------------------------
    # Run forward passes
    # ----------------------------------------------------------------
    print("\n[4/4] Running forward passes...")

    # We need to run the full prompt (all 6 tokens) through the model for
    # proper attention context. For the float and hw models, we use prefill.
    # For INT8, we run token by token with KV cache matching hardware.

    # --- Float model: full prefill, extract position 5 ---
    # Run full prefill on "ROMEO:" and extract activations for position 5
    with torch.no_grad():
        # For accurate comparison we need single-token activations at pos 5
        # with all prior context. Use full sequence and slice position 5.
        prompt_tokens = torch.tensor([[ord(c) for c in prompt_str]])
        # Full prefill
        logits_float, _ = model_float(prompt_tokens)

    # For detailed per-stage float activations, we'll manually trace through
    float_acts = {}
    with torch.no_grad():
        tokens_t = torch.tensor([[ord(c) for c in prompt_str]])
        positions = torch.arange(0, len(prompt_str))

        x = model_float.tok_emb(tokens_t) + model_float.pos_emb(positions)
        float_acts["embedding"] = x[0, target_pos].cpu().numpy()

        for layer_idx, layer in enumerate(model_float.layers):
            h = layer.attn_norm(x)
            float_acts[f"L{layer_idx}_attn_norm"] = h[0, target_pos].cpu().numpy()

            # QKV
            q = layer.attn.q_proj(h)
            float_acts[f"L{layer_idx}_q_proj"] = q[0, target_pos].cpu().numpy()
            k = layer.attn.k_proj(h)
            float_acts[f"L{layer_idx}_k_proj"] = k[0, target_pos].cpu().numpy()
            v = layer.attn.v_proj(h)
            float_acts[f"L{layer_idx}_v_proj"] = v[0, target_pos].cpu().numpy()

            attn_out, _ = layer.attn(h, mask=True, kv_cache=None)
            float_acts[f"L{layer_idx}_attn_out"] = attn_out[0, target_pos].cpu().numpy()

            x = x + attn_out
            float_acts[f"L{layer_idx}_attn_residual"] = x[0, target_pos].cpu().numpy()

            h = layer.ff_norm(x)
            float_acts[f"L{layer_idx}_ff_norm"] = h[0, target_pos].cpu().numpy()

            up = layer.ff.up(h)
            float_acts[f"L{layer_idx}_ff_up_raw"] = up[0, target_pos].cpu().numpy()
            up_relu = F.relu(up)
            float_acts[f"L{layer_idx}_ff_up_relu"] = up_relu[0, target_pos].cpu().numpy()
            down = layer.ff.down(up_relu)
            float_acts[f"L{layer_idx}_ff_down"] = down[0, target_pos].cpu().numpy()

            x = x + down
            float_acts[f"L{layer_idx}_ff_residual"] = x[0, target_pos].cpu().numpy()

        x_norm = model_float.final_norm(x)
        float_acts["final_norm"] = x_norm[0, target_pos].cpu().numpy()
        logits_f = model_float.output_head(x_norm) * (config_float.d_model ** -0.5)
        float_acts["logits"] = logits_f[0, target_pos].cpu().numpy()

    # --- HW-mode model: full prefill, extract position 5 ---
    hw_acts = {}
    with torch.no_grad():
        tokens_t = torch.tensor([[ord(c) for c in prompt_str]])
        positions = torch.arange(0, len(prompt_str))

        tok_raw = model_hw.tok_emb(tokens_t)
        pos_raw = model_hw.pos_emb(positions)
        tok_q = ste_round_clamp(tok_raw * 1024, -32768, 32767)
        pos_q = ste_round_clamp(pos_raw * 1024, -32768, 32767)
        x = ste_shift_right(tok_q + pos_q, 3)
        x = ste_hw_int8(x)
        hw_acts["embedding"] = x[0, target_pos].cpu().numpy()

        for layer_idx, layer in enumerate(model_hw.layers):
            h = layer.attn_norm(x)
            hw_acts[f"L{layer_idx}_attn_norm"] = h[0, target_pos].cpu().numpy()

            q = layer.attn.q_proj(h)
            hw_acts[f"L{layer_idx}_q_proj"] = q[0, target_pos].cpu().numpy()
            k = layer.attn.k_proj(h)
            hw_acts[f"L{layer_idx}_k_proj"] = k[0, target_pos].cpu().numpy()
            v = layer.attn.v_proj(h)
            hw_acts[f"L{layer_idx}_v_proj"] = v[0, target_pos].cpu().numpy()

            attn_out, _ = layer.attn(h, mask=True, kv_cache=None)
            hw_acts[f"L{layer_idx}_attn_out"] = attn_out[0, target_pos].cpu().numpy()

            x = x + attn_out
            x = ste_hw_int8(x)
            hw_acts[f"L{layer_idx}_attn_residual"] = x[0, target_pos].cpu().numpy()

            h = layer.ff_norm(x)
            hw_acts[f"L{layer_idx}_ff_norm"] = h[0, target_pos].cpu().numpy()

            up = layer.ff.up(h)
            hw_acts[f"L{layer_idx}_ff_up_raw"] = up[0, target_pos].cpu().numpy()
            up_relu = F.relu(up)
            up_relu = ste_round_clamp_int8(up_relu)
            hw_acts[f"L{layer_idx}_ff_up_relu"] = up_relu[0, target_pos].cpu().numpy()
            down = layer.ff.down(up_relu)
            hw_acts[f"L{layer_idx}_ff_down"] = down[0, target_pos].cpu().numpy()

            x = x + down
            x = ste_hw_int8(x)
            hw_acts[f"L{layer_idx}_ff_residual"] = x[0, target_pos].cpu().numpy()

        x_norm = model_hw.final_norm(x)
        hw_acts["final_norm"] = x_norm[0, target_pos].cpu().numpy()
        emb_q = ste_round_clamp(model_hw.tok_emb.weight * 1024, -32768, 32767)
        logits_raw = torch.matmul(x_norm, emb_q.T)
        logits_hw = logits_raw / (64.0 * 1024.0 * math.sqrt(config_hw.d_model))
        hw_acts["logits"] = logits_hw[0, target_pos].cpu().numpy()
        hw_acts["logits_raw_int24"] = logits_raw[0, target_pos].cpu().numpy()

    # --- INT8 reference: run full prompt ---
    print("  Running INT8 reference inference (token by token)...")
    int8_acts = int8_forward_single_step(target_token, target_pos, int8_weights, int8_config)

    # ================================================================
    # STAGE COMPARISONS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  PER-STAGE ACTIVATION COMPARISON")
    print("#" * 72)

    stages = [
        "embedding",
        "L0_attn_norm", "L0_q_proj", "L0_k_proj", "L0_v_proj",
        "L0_attn_out", "L0_attn_residual",
        "L0_ff_norm", "L0_ff_up_raw", "L0_ff_up_relu", "L0_ff_down",
        "L0_ff_residual",
        "L1_attn_norm", "L1_q_proj", "L1_k_proj", "L1_v_proj",
        "L1_attn_out", "L1_attn_residual",
        "L1_ff_norm", "L1_ff_up_raw", "L1_ff_up_relu", "L1_ff_down",
        "L1_ff_residual",
        "final_norm",
        "logits",
    ]

    all_stats = []
    for stage in stages:
        if stage not in float_acts:
            print(f"\n  SKIP: {stage} (not in float_acts)")
            continue
        if stage not in hw_acts:
            print(f"\n  SKIP: {stage} (not in hw_acts)")
            continue
        if stage not in int8_acts:
            print(f"\n  SKIP: {stage} (not in int8_acts)")
            continue

        stats = report_stage(stage, float_acts[stage], hw_acts[stage], int8_acts[stage])
        all_stats.append(stats)

    # ================================================================
    # ATTENTION DETAIL
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  ATTENTION SCORE & PROBABILITY DETAIL")
    print("#" * 72)

    for layer_idx in range(2):
        key_scores = f"L{layer_idx}_attn_scores"
        key_probs = f"L{layer_idx}_attn_probs"
        if key_scores in int8_acts:
            for h in range(config_hw.n_heads):
                scores = int8_acts[key_scores][h]
                probs = int8_acts[key_probs][h]
                print(f"\n  Layer {layer_idx} Head {h} INT8 attention at pos {target_pos}:")
                print(f"    Scores: {scores}")
                print(f"    Probs:  {probs}")
                print(f"    Max prob: {max(probs)}, Sum: {sum(probs)}")

    # ================================================================
    # SUMMARY: INFORMATION LOSS RANKING
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  INFORMATION LOSS RANKING (sorted by cosine similarity drop)")
    print("#" * 72)

    print(f"\n  {'Stage':<25s} {'cos(F,HW)':>10s} {'cos(F,I8)':>10s}"
          f" {'HW_entropy':>11s} {'HW_uniq':>8s} {'HW_range%':>10s}"
          f" {'HW_sat%':>8s}")
    print("  " + "-" * 82)

    # Sort by cos(float,hw) ascending -- worst first
    ranked = sorted(all_stats, key=lambda s: s["cos_float_hw"])
    for s in ranked:
        print(f"  {s['name']:<25s} {s['cos_float_hw']:10.6f} {s['cos_float_int8']:10.6f}"
              f" {s['hw_entropy']:11.2f} {s['hw_unique']:8d} {s['hw_range_util']*100:10.1f}"
              f" {s['hw_saturated_pct']:8.1f}")

    # ================================================================
    # LOGIT ANALYSIS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  LOGIT ANALYSIS (top-10 predictions)")
    print("#" * 72)

    float_logits = float_acts["logits"]
    hw_logits = hw_acts["logits"]
    int8_logits = int8_acts["logits"]

    # Float top 10
    float_top = np.argsort(float_logits)[::-1][:10]
    print(f"\n  FLOAT model top-10 predictions:")
    for i, idx in enumerate(float_top):
        c = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
        print(f"    #{i+1}: token={idx:3d} '{c}' logit={float_logits[idx]:.4f}")

    # HW top 10
    hw_top = np.argsort(hw_logits)[::-1][:10]
    print(f"\n  HW_MODE model top-10 predictions:")
    for i, idx in enumerate(hw_top):
        c = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
        print(f"    #{i+1}: token={idx:3d} '{c}' logit={hw_logits[idx]:.4f}")

    # INT8 top 10 (raw INT24)
    int8_top = np.argsort(int8_logits)[::-1][:10]
    print(f"\n  INT8 reference top-10 predictions:")
    for i, idx in enumerate(int8_top):
        c = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
        print(f"    #{i+1}: token={idx:3d} '{c}' logit={int8_logits[idx]:.0f}")

    # Logit distribution analysis
    print(f"\n  Logit distribution (float vs hw vs int8):")
    for name, logits in [("FLOAT", float_logits), ("HW", hw_logits), ("INT8", int8_logits)]:
        top1 = np.max(logits)
        top2 = np.sort(logits)[-2]
        print(f"    {name:5s}: top1={top1:.2f}  top2={top2:.2f}"
              f"  gap={top1-top2:.4f}  std={np.std(logits):.4f}"
              f"  unique={len(np.unique(np.round(logits, 2)))}")

    # ================================================================
    # TDot >>4 SHIFT ANALYSIS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  TDot >>4 SHIFT ANALYSIS (pre-shift accumulator range)")
    print("#" * 72)
    print("  Checking if >>4 is too aggressive by looking at ternary dot product")
    print("  accumulators before the shift...")
    print()

    for layer_idx in range(2):
        prefix = f"layers.{layer_idx}"
        # Get the normalized input for this layer from INT8
        norm_key = f"L{layer_idx}_attn_norm"
        if norm_key in int8_acts:
            normed = int8_acts[norm_key]
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                w_key = f"{prefix}.attn.{proj_name}"
                if w_key in int8_weights:
                    w = int8_weights[w_key]
                    d_out, d_in = w.shape
                    # Compute raw accumulators (before >>4)
                    raw_accs = []
                    if proj_name == "o_proj":
                        # o_proj uses attn output, not normed
                        inp = int8_acts.get(f"L{layer_idx}_attn_out_pre_oproj", normed)
                    else:
                        inp = normed
                    for o in range(d_out):
                        acc = 0
                        for j in range(d_in):
                            acc += int(inp[j]) * int(w[o, j])
                        raw_accs.append(acc)
                    raw_np = np.array(raw_accs)
                    shifted = raw_np >> 4
                    clamped = np.clip(shifted, -128, 127)
                    n_clipped = int(np.sum(np.abs(shifted) > 127))
                    print(f"  L{layer_idx}.{proj_name}: raw_acc range=[{raw_np.min()}, {raw_np.max()}]"
                          f"  >>4=[{shifted.min()}, {shifted.max()}]"
                          f"  clipped={n_clipped}/{d_out}"
                          f"  raw_std={raw_np.std():.1f}  >>4_std={shifted.std():.2f}")

        # FFN analysis
        ff_norm_key = f"L{layer_idx}_ff_norm"
        if ff_norm_key in int8_acts:
            normed_ff = int8_acts[ff_norm_key]
            # FFN up (64->256)
            w_up = int8_weights[f"{prefix}.ff.up"]
            d_out_up, d_in_up = w_up.shape
            raw_up = []
            for o in range(d_out_up):
                acc = 0
                for j in range(d_in_up):
                    acc += int(normed_ff[j]) * int(w_up[o, j])
                raw_up.append(acc)
            raw_up_np = np.array(raw_up)
            shifted_up = raw_up_np >> 4
            n_clip_up = int(np.sum(np.abs(shifted_up) > 127))
            print(f"  L{layer_idx}.ff.up:    raw_acc range=[{raw_up_np.min()}, {raw_up_np.max()}]"
                  f"  >>4=[{shifted_up.min()}, {shifted_up.max()}]"
                  f"  clipped={n_clip_up}/{d_out_up}"
                  f"  raw_std={raw_up_np.std():.1f}  >>4_std={shifted_up.std():.2f}")

            # FFN down (256->64, multi-pass)
            relu_in = int8_acts[f"L{layer_idx}_ff_up_relu"]
            w_down = int8_weights[f"{prefix}.ff.down"]
            d_out_down, d_in_down = w_down.shape
            n_passes = d_in_down // config_hw.d_model
            raw_down = []
            for o in range(d_out_down):
                acc = 0
                for p in range(n_passes):
                    partial = 0
                    for j in range(config_hw.d_model):
                        idx = p * config_hw.d_model + j
                        partial += int(relu_in[idx]) * int(w_down[o, idx])
                    partial = to_sint(partial, 24)
                    if p == 0:
                        acc = partial
                    else:
                        acc = to_sint(acc + partial, 24)
                raw_down.append(acc)
            raw_down_np = np.array(raw_down)
            shifted_down = raw_down_np >> 4
            n_clip_down = int(np.sum(np.abs(shifted_down) > 127))
            print(f"  L{layer_idx}.ff.down:  raw_acc range=[{raw_down_np.min()}, {raw_down_np.max()}]"
                  f"  >>4=[{shifted_down.min()}, {shifted_down.max()}]"
                  f"  clipped={n_clip_down}/{d_out_down}"
                  f"  raw_std={raw_down_np.std():.1f}  >>4_std={shifted_down.std():.2f}")

    # ================================================================
    # RMSNORM ANALYSIS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  RMSNORM LUT ANALYSIS")
    print("#" * 72)

    for layer_idx in range(2):
        for norm_type, input_key in [("attn_norm", f"L{layer_idx}_attn_residual" if layer_idx > 0 else "embedding"),
                                      ("ff_norm", f"L{layer_idx}_attn_residual")]:
            if layer_idx == 0 and norm_type == "attn_norm":
                input_key = "embedding"

            if input_key not in int8_acts:
                continue

            inp = int8_acts[input_key]
            sum_sq = sum(int(x)**2 for x in inp)
            mean_sq = (sum_sq >> 6) & 0xFFFF
            lut_idx = (mean_sq >> 6) & 0xFF
            inv_rms = INV_SQRT_LUT[lut_idx]

            gamma_key = f"layer{layer_idx}_{norm_type}"
            gamma = int8_weights[gamma_key]
            gamma_stats = f"min={int(gamma.min())}, max={int(gamma.max())}, std={gamma.astype(float).std():.1f}"

            output_key = f"L{layer_idx}_{norm_type}"
            output = int8_acts[output_key]

            print(f"\n  L{layer_idx}.{norm_type}:")
            print(f"    Input: sum_sq={sum_sq}, mean_sq={mean_sq}, lut_idx={lut_idx}, inv_rms={inv_rms}")
            print(f"    Input range: [{inp.min()}, {inp.max()}], std={inp.astype(float).std():.2f}")
            print(f"    Gamma Q5.10: {gamma_stats}")
            print(f"    Output range: [{output.min()}, {output.max()}], std={output.astype(float).std():.2f}")
            print(f"    Scale factor: inv_rms/256 = {inv_rms/256:.4f} (ideally sqrt(64)/rms)")
            if sum_sq > 0:
                ideal_inv_rms = math.sqrt(64.0 * 16384.0**2 / sum_sq)
                print(f"    Ideal inv_rms = {ideal_inv_rms:.1f}, LUT gives {inv_rms}"
                      f" (error = {abs(inv_rms - ideal_inv_rms)/ideal_inv_rms*100:.1f}%)")

    # Final norm
    final_input = int8_acts["L1_ff_residual"]
    sum_sq = sum(int(x)**2 for x in final_input)
    mean_sq = (sum_sq >> 6) & 0xFFFF
    lut_idx = (mean_sq >> 6) & 0xFF
    inv_rms = INV_SQRT_LUT[lut_idx]
    gamma = int8_weights["final_norm"]
    print(f"\n  final_norm:")
    print(f"    Input: sum_sq={sum_sq}, mean_sq={mean_sq}, lut_idx={lut_idx}, inv_rms={inv_rms}")
    print(f"    Input range: [{final_input.min()}, {final_input.max()}],"
          f" std={final_input.astype(float).std():.2f}")
    print(f"    Gamma Q5.10: min={int(gamma.min())}, max={int(gamma.max())},"
          f" std={gamma.astype(float).std():.1f}")
    output = int8_acts["final_norm"]
    print(f"    Output range: [{output.min()}, {output.max()}],"
          f" std={output.astype(float).std():.2f}")

    # ================================================================
    # CUMULATIVE INFORMATION LOSS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  CUMULATIVE COSINE SIMILARITY (float vs hw at each stage)")
    print("#" * 72)
    print(f"\n  {'Stage':<25s} {'cos(F,HW)':>10s}  {'delta':>8s}  Notes")
    print("  " + "-" * 70)

    prev_cos = 1.0
    for s in all_stats:
        delta = s["cos_float_hw"] - prev_cos
        notes = []
        if s["hw_saturated_pct"] > 5:
            notes.append(f"SATURATED={s['hw_saturated_pct']:.0f}%")
        if s["hw_range_util"] < 0.2:
            notes.append(f"LOW_RANGE={s['hw_range_util']*100:.0f}%")
        if s["hw_unique"] < 10:
            notes.append(f"FEW_UNIQUE={s['hw_unique']}")
        note_str = ", ".join(notes) if notes else ""
        print(f"  {s['name']:<25s} {s['cos_float_hw']:10.6f}  {delta:+8.6f}  {note_str}")
        prev_cos = s["cos_float_hw"]

    # ================================================================
    # WEIGHT ANALYSIS
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  WEIGHT SPARSITY AND DISTRIBUTION")
    print("#" * 72)

    # Check float model weight magnitudes vs ternary
    for layer_idx in range(2):
        prefix = f"layers.{layer_idx}"
        for proj in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                      "ff.up", "ff.down"]:
            # Float model weights
            w_float = cp1["model"][f"{prefix}.{proj}.weight"].numpy()
            w_hw = cp2["model"][f"{prefix}.{proj}.weight"].numpy()
            # Ternary quantization
            w_tern_float, scale_float = ternary_quantize(torch.tensor(w_float))
            w_tern_hw, scale_hw = ternary_quantize(torch.tensor(w_hw))
            w_tern_float = w_tern_float.numpy()
            w_tern_hw = w_tern_hw.numpy()

            nonzero_float = np.count_nonzero(w_tern_float)
            nonzero_hw = np.count_nonzero(w_tern_hw)
            total = w_float.size
            agree = np.sum(w_tern_float == w_tern_hw)

            print(f"  L{layer_idx}.{proj}: "
                  f"float_scale={float(scale_float):.4f} hw_scale={float(scale_hw):.4f} "
                  f"nonzero_f={nonzero_float}/{total}({nonzero_float/total*100:.0f}%) "
                  f"nonzero_hw={nonzero_hw}/{total}({nonzero_hw/total*100:.0f}%) "
                  f"ternary_agree={agree}/{total}({agree/total*100:.0f}%)")

    # ================================================================
    # DIAGNOSIS SUMMARY
    # ================================================================
    print("\n\n" + "#" * 72)
    print("#  DIAGNOSIS SUMMARY")
    print("#" * 72)

    # Find the biggest cosine sim drops
    if len(all_stats) >= 2:
        drops = []
        for i in range(1, len(all_stats)):
            drop = all_stats[i]["cos_float_hw"] - all_stats[i-1]["cos_float_hw"]
            drops.append((all_stats[i]["name"], drop, all_stats[i]))

        worst_drops = sorted(drops, key=lambda x: x[1])[:5]
        print("\n  Biggest cosine similarity drops (float vs hw):")
        for name, drop, s in worst_drops:
            print(f"    {name}: {drop:+.6f}"
                  f" (sat={s['hw_saturated_pct']:.0f}%,"
                  f" range={s['hw_range_util']*100:.0f}%,"
                  f" unique={s['hw_unique']})")

    # Check if "the" dominates
    the_tokens = [ord('t'), ord('h'), ord('e')]
    print(f"\n  Token 't'={ord('t')}, 'h'={ord('h')}, 'e'={ord('e')}")
    for name, logits in [("FLOAT", float_acts["logits"]),
                          ("HW", hw_acts["logits"]),
                          ("INT8", int8_acts["logits"])]:
        top_idx = np.argmax(logits)
        top_char = chr(top_idx) if 32 <= top_idx < 127 else f"\\x{top_idx:02x}"
        the_logits = [logits[t] for t in the_tokens]
        print(f"    {name:5s}: top='{top_char}'({top_idx})  "
              f"t={logits[ord('t')]:.2f} h={logits[ord('h')]:.2f} e={logits[ord('e')]:.2f}"
              f"  top_logit={logits[top_idx]:.2f}")


if __name__ == "__main__":
    main()
