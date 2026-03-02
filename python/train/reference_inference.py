"""Bit-accurate INT8 reference inference for RTL validation.

This module reimplements the model forward pass using only integer
arithmetic matching the FPGA hardware datapath exactly. Used to generate
golden test vectors for SpinalHDL simulation.

Every operation matches the hardware:
- Embedding: INT16 tok + INT16 pos, >> 3 -> INT8 (saturating clamp)
- RMSNorm: 256-entry inv_sqrt LUT (Q2.14), (x*invRms>>8)*gamma>>10, clamp
- TDot projections: ternary dot product -> INT24, >> 4 -> INT8 (saturating clamp)
- Attention scores: INT8 x INT8 -> INT24, *45 >> 8 -> INT16
- Softmax: PL-exp (4 segments) + 256-entry reciprocal LUT
- Attention value: UINT8 reinterpreted as SInt8 * INT8, >> 8 -> INT8 (saturating clamp)
- Residual: clamp(x + result, -128, 127)
- FFN: TDot up >> 4, ReLU, TDot down accumulated >> 4
- Logits: INT8 query * INT16 embedding -> INT24 accumulated
"""

import json
import math
import os

import numpy as np

from .config import ZyboGPTConfig


FRAC_BITS = 10  # Q5.10 fixed-point


# ================================================================
# Galois LFSR (matches hardware SamplingUnit.scala and Rust sampling.rs)
# ================================================================

def lfsr_step(state: int) -> int:
    """Galois LFSR with polynomial 0xD000_0001.

    bit = state & 1; state >>= 1; if bit: state ^= 0xD000_0001
    """
    bit = state & 1
    state >>= 1
    if bit:
        state ^= 0xD0000001
    return state & 0xFFFFFFFF


# ================================================================
# Bit-accurate integer helpers
# ================================================================

def to_sint(val: int, bits: int) -> int:
    """Truncate to signed integer of given bit width (like SpinalHDL .resize())."""
    mask = (1 << bits) - 1
    val = int(val) & mask
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def clamp_int8(val: int) -> int:
    """Saturating clamp to signed INT8 [-128, 127].

    Used where hardware uses saturating arithmetic instead of truncation.
    """
    return max(-128, min(127, int(val)))


def to_uint(val: int, bits: int) -> int:
    """Truncate to unsigned integer of given bit width."""
    return int(val) & ((1 << bits) - 1)


# ================================================================
# Hardware LUT builders (match RMSNorm.scala and Softmax.scala)
# ================================================================

def _build_inv_sqrt_lut() -> list[int]:
    """Build the 256-entry inv_sqrt LUT matching RMSNorm.scala.

    LUT[i] = round(16384 / sqrt(i * 64 + 32)) for i >= 1
    LUT[0] = 16383 (max, protect div-by-zero)
    Values are Q2.14 unsigned (0-16383).
    """
    lut_shift = 6
    lut_frac = 14
    lut = []
    for i in range(256):
        if i == 0:
            lut.append(16383)
        else:
            mean_sq_approx = i * (1 << lut_shift) + (1 << (lut_shift - 1))
            inv_sqrt = (1 << lut_frac) / math.sqrt(mean_sq_approx)
            lut.append(max(0, min(16383, round(inv_sqrt))))
    return lut


def _build_recip_lut() -> list[int]:
    """Build the 256-entry reciprocal LUT matching Softmax.scala.

    Index by expSum >> 7.
    LUT[i] = round(65536 / (i * 128 + 64)) for i >= 1
    LUT[0] = 65535 (max, protect div-by-zero)
    """
    recip_shift = 7
    lut = []
    for i in range(256):
        if i == 0:
            lut.append(65535)
        else:
            exp_sum_approx = i * (1 << recip_shift) + (1 << (recip_shift - 1))
            recip_val = 65536.0 / exp_sum_approx
            lut.append(max(0, min(65535, round(recip_val))))
    return lut


INV_SQRT_LUT = _build_inv_sqrt_lut()
RECIP_LUT = _build_recip_lut()


# ================================================================
# Weight loading
# ================================================================

def load_weights(export_dir: str) -> dict:
    """Load exported weights into numpy arrays."""
    with open(os.path.join(export_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    weights = {}

    # Load ternary weights
    ternary_data = np.fromfile(os.path.join(export_dir, "weights_ternary.bin"), dtype=np.uint8)
    for name, info in meta["ternary_layers"].items():
        offset = info["bram_offset"]
        length = info["packed_bytes"]
        packed = ternary_data[offset : offset + length]
        shape = info["shape"]
        trits = unpack_ternary_5per_byte(packed, np.prod(shape))
        weights[name] = trits.reshape(shape)
        weights[f"{name}_scale"] = info["scale_int16"]

    # Load full-precision weights
    full_data = np.fromfile(os.path.join(export_dir, "weights_full.bin"), dtype=np.int16)
    config = ZyboGPTConfig(**{k: v for k, v in meta["config"].items() if k != "head_dim"})

    idx = 0
    # tok_emb: (vocab, d_model)
    size = config.vocab_size * config.d_model
    weights["tok_emb"] = full_data[idx : idx + size].reshape(config.vocab_size, config.d_model)
    idx += size

    # pos_emb: (ctx_len, d_model)
    size = config.ctx_len * config.d_model
    weights["pos_emb"] = full_data[idx : idx + size].reshape(config.ctx_len, config.d_model)
    idx += size

    # Norm weights
    for i in range(config.n_layers):
        for norm_name in ["attn_norm", "ff_norm"]:
            key = f"layer{i}_{norm_name}"
            weights[key] = full_data[idx : idx + config.d_model].copy()
            idx += config.d_model

    weights["final_norm"] = full_data[idx : idx + config.d_model].copy()

    return weights, config


def unpack_ternary_5per_byte(packed: np.ndarray, num_trits: int) -> np.ndarray:
    """Unpack 5 ternary trits per byte back to {-1, 0, +1}."""
    trits = []
    for byte in packed:
        b = int(byte)
        for _ in range(5):
            trits.append((b % 3) - 1)  # {0,1,2} -> {-1,0,+1}
            b //= 3
            if len(trits) >= num_trits:
                break
        if len(trits) >= num_trits:
            break
    return np.array(trits[:num_trits], dtype=np.int8)


# ================================================================
# Bit-accurate ternary dot product (matches TDot + storeTdotResults)
# ================================================================

def hw_ternary_matvec(x_int8: np.ndarray, w_ternary: np.ndarray) -> np.ndarray:
    """Ternary matrix-vector multiply matching hardware TDot + >> 4 + clamp.

    x_int8: (d_in,) INT8
    w_ternary: (d_out, d_in) in {-1,0,+1}
    returns: (d_out,) INT8 -- each result is clamp(dot_product >> 4, -128, 127)

    Uses >> 4 (not >> 8) to preserve dynamic range. With ~65% nonzero ternary
    weights, raw accumulator is ~+-700, >> 4 gives ~+-44 (good INT8 utilization).
    """
    d_out, d_in = w_ternary.shape
    result = np.zeros(d_out, dtype=np.int8)
    for o in range(d_out):
        acc = 0
        for j in range(d_in):
            acc += int(x_int8[j]) * int(w_ternary[o, j])
        # Hardware: INT24 accumulator >> 4, saturate to INT8
        acc = to_sint(acc, 24)
        result[o] = clamp_int8(acc >> 4)
    return result


def hw_ternary_matvec_accum(x_int8: np.ndarray, w_ternary: np.ndarray,
                             d_in_slice: int) -> np.ndarray:
    """Ternary matvec with multi-pass accumulation (for FFN down projection).

    When input is wider than TDot width, hardware accumulates partial sums
    across passes, then >> 4 at the end.

    x_int8: (d_in,) INT8, where d_in may be > d_in_slice
    w_ternary: (d_out, d_in) in {-1,0,+1}
    d_in_slice: width of each TDot pass (= d_model = 64)
    returns: (d_out,) INT8
    """
    d_out, d_in = w_ternary.shape
    n_passes = d_in // d_in_slice
    result = np.zeros(d_out, dtype=np.int8)

    for o in range(d_out):
        acc = 0
        for p in range(n_passes):
            partial = 0
            for j in range(d_in_slice):
                idx = p * d_in_slice + j
                partial += int(x_int8[idx]) * int(w_ternary[o, idx])
            partial = to_sint(partial, 24)
            if p == 0:
                acc = partial
            else:
                acc = to_sint(acc + partial, 24)
        result[o] = clamp_int8(acc >> 4)
    return result


# ================================================================
# Hardware RMSNorm (matches RMSNorm.scala exactly)
# ================================================================

def hw_rmsnorm(x_int8: np.ndarray, gamma_int16: np.ndarray) -> np.ndarray:
    """Bit-accurate RMSNorm matching hardware LUT-based implementation.

    Computes: y[i] = clamp((x[i] * invRms >> 8) * gamma[i] >> 10, -128, 127)
    The effective total shift is 18 (not 24), providing a 64x (2^6) scale-up
    so the output fills the INT8 range for downstream TDot operations.

    x_int8: (d_model,) INT8
    gamma_int16: (d_model,) INT16 Q5.10
    returns: (d_model,) INT8
    """
    dim = len(x_int8)
    lut_shift = 6
    shift2 = 10  # = (lutFrac + fracBits) - 8 - 6 = 24 - 8 - 6

    # SUM_SQ state: accumulate x[i]^2
    sum_sq = 0
    for i in range(dim):
        sq = int(x_int8[i]) * int(x_int8[i])  # always positive
        sum_sq += sq

    # Compute LUT index: meanSq = sumSq >> log2(dim), lutIdx = meanSq >> lutShift
    mean_sq = to_uint(sum_sq >> 6, 16)  # dim=64=2^6
    lut_idx = to_uint(mean_sq >> lut_shift, 8)
    inv_rms_val = INV_SQRT_LUT[lut_idx]

    # SCALE state: y[i] = clamp((x[i] * invRms >> 8) * gamma[i] >> 10, -128, 127)
    result = np.zeros(dim, dtype=np.int8)
    for i in range(dim):
        x_val = to_sint(int(x_int8[i]), 32)
        inv_rms = to_sint(inv_rms_val, 32)  # .asSInt.resize(32) -- always positive
        gamma_val = to_sint(int(gamma_int16[i]), 32)

        prod1 = x_val * inv_rms
        prod1_trunc = to_sint(prod1 >> 8, 32)
        prod2 = prod1_trunc * gamma_val
        shifted = to_sint(prod2 >> shift2, 16)

        result[i] = max(-128, min(127, shifted))

    return result


# ================================================================
# Hardware PL-exp and Softmax (matches Softmax.scala exactly)
# ================================================================

def hw_pl_exp(x: int) -> int:
    """Piecewise-linear exp approximation matching Softmax.scala plExp().

    Input: shifted score (INT16, always <= 0)
    Output: approximate exp value (UINT16, max 256 for input 0)
    """
    # Clamp to [-128, 0]
    if x >= 0:
        x = 0
    elif x < -128:
        x = -128

    if x >= -3:
        # 256 + x*64 -> x=0: 256, x=-3: 64
        return max(0, 256 + x * 64)
    elif x >= -8:
        # 64 + (x+3)*11 -> x=-3: 64, x=-8: 9
        return max(0, 64 + (x + 3) * 11)
    elif x >= -24:
        # (x+24) as unsigned -> x=-24: 0, x=-9: 15
        return max(0, x + 24)
    else:
        return 0


def hw_softmax(scores_int16: list[int], length: int, ctx_len: int = 128) -> list[int]:
    """Bit-accurate softmax matching Softmax.scala.

    scores_int16: list of INT16 scores (ctx_len entries)
    length: number of valid positions
    ctx_len: max context length
    returns: list of UINT8 probabilities (ctx_len entries)
    """
    # FIND_MAX: scan for maximum over valid positions
    max_val = -32767
    for i in range(min(length, ctx_len)):
        if scores_int16[i] > max_val:
            max_val = scores_int16[i]

    # SUBTRACT_EXP: compute PL-exp of (score - max)
    exp_vals = []
    for i in range(ctx_len):
        if i < length:
            shifted = to_sint(scores_int16[i] - max_val, 16)
            exp_vals.append(hw_pl_exp(shifted))
        else:
            exp_vals.append(0)

    # SUM_EXP
    exp_sum = to_uint(sum(exp_vals), 24)
    exp_sum = max(exp_sum, 1)

    # RECIP: LUT lookup
    recip_addr = to_uint(exp_sum >> 7, 8)
    recip_val = RECIP_LUT[recip_addr]

    # NORMALIZE: prob[i] = clamp((exp[i] * recipVal) >> 8, 0, 255)
    probs = []
    for i in range(ctx_len):
        product = to_uint(exp_vals[i], 32) * to_uint(recip_val, 32)
        prob = to_uint(product >> 8, 16)
        probs.append(min(prob, 255))

    return probs


# ================================================================
# Main bit-accurate inference (matches FPGA datapath)
# ================================================================

def reference_inference(
    tokens: list[int],
    weights: dict,
    config: ZyboGPTConfig,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    seed: int = 0xDEADBEEF,
) -> tuple[list[int], list[dict]]:
    """Run full inference matching FPGA hardware datapath exactly.

    Single-token incremental inference with KV cache, matching:
    Sequencer (IDLE->EMBED->LAYER_LOOP->FINAL_NORM->OUTPUT_LOGITS->ARGMAX->DONE)

    If temperature > 0, uses temperature sampling with Galois LFSR for
    deterministic randomness, bit-accurate with hardware SamplingUnit.
    """
    debug_data = []
    d = config.d_model
    n_h = config.n_heads
    h_d = config.head_dim

    # Temperature sampling setup
    if temperature > 0:
        inv_temp = 65536 // round(temperature * 256)
    else:
        inv_temp = 0
    lfsr_state = seed & 0xFFFFFFFF

    # KV cache: per layer, per head, list of (k_vec, v_vec) per position
    # k_vec and v_vec are INT8 arrays of length head_dim
    kv_cache = [[[] for _ in range(n_h)] for _ in range(config.n_layers)]

    # Process prompt tokens first, then generate
    all_tokens = list(tokens)

    for step in range(len(tokens) + max_new_tokens):
        if step < len(tokens):
            token = tokens[step]
            position = step
        else:
            token = all_tokens[-1]
            position = step

        step_debug = {"step": step, "token_in": token, "position": position}

        # ===========================================================
        # EMBED: tok_emb[token] + pos_emb[position], >> 3 -> INT8
        # (Sequencer.scala line 122-124 + Embedding.scala)
        # ===========================================================
        act = np.zeros(d, dtype=np.int8)
        for i in range(d):
            tok_val = int(weights["tok_emb"][token, i])  # INT16
            pos_val = int(weights["pos_emb"][position, i])  # INT16
            emb_sum = tok_val + pos_val  # INT16 + INT16
            # >> (fracBits - 7) = >> 3, saturate to INT8
            act[i] = clamp_int8(emb_sum >> 3)

        step_debug["after_embed"] = act.tolist()

        # ===========================================================
        # LAYER_LOOP: process through n_layers transformer layers
        # ===========================================================
        for layer_idx in range(config.n_layers):
            prefix = f"layers.{layer_idx}"

            # --- ATTN_NORM: RMSNorm ---
            gamma = weights[f"layer{layer_idx}_attn_norm"]
            normed = hw_rmsnorm(act, gamma)
            step_debug[f"after_layer{layer_idx}_attn_norm"] = normed.tolist()

            # --- QKV Projections via TDot (each >> 8 -> INT8) ---
            q = hw_ternary_matvec(normed, weights[f"{prefix}.attn.q_proj"])
            step_debug[f"after_layer{layer_idx}_q_proj"] = q.tolist()
            k = hw_ternary_matvec(normed, weights[f"{prefix}.attn.k_proj"])
            step_debug[f"after_layer{layer_idx}_k_proj"] = k.tolist()
            v = hw_ternary_matvec(normed, weights[f"{prefix}.attn.v_proj"])
            step_debug[f"after_layer{layer_idx}_v_proj"] = v.tolist()

            # --- STORE_KV: store K,V into cache per head ---
            for h in range(n_h):
                k_head = k[h * h_d : (h + 1) * h_d].copy()
                v_head = v[h * h_d : (h + 1) * h_d].copy()
                if position < len(kv_cache[layer_idx][h]):
                    kv_cache[layer_idx][h][position] = (k_head, v_head)
                else:
                    kv_cache[layer_idx][h].append((k_head, v_head))

            # --- ATTN_SCORE + SCALE_MASK + SOFTMAX + ATTN_VALUE per head ---
            attn_out = np.zeros(d, dtype=np.int8)

            for h in range(n_h):
                # ATTN_SCORE: Q[head] . K[pos] for each cached position
                scores = [0] * config.ctx_len
                for pos in range(position + 1):
                    k_cached = kv_cache[layer_idx][h][pos][0]
                    # 32-element dot product (INT8 x INT8 -> INT24)
                    dot = 0
                    for i in range(h_d):
                        dot += int(q[h * h_d + i]) * int(k_cached[i])
                    dot = to_sint(dot, 24)
                    # Scale: *45 >> 8, truncate to INT16
                    scores[pos] = to_sint((dot * 45) >> 8, 16)

                # Causal mask: future positions = -32767
                for i in range(position + 1, config.ctx_len):
                    scores[i] = -32767

                # SOFTMAX
                probs = hw_softmax(scores, position + 1, config.ctx_len)

                # ATTN_VALUE: weighted sum
                acc = [0] * h_d
                for pos in range(position + 1):
                    v_cached = kv_cache[layer_idx][h][pos][1]
                    prob_uint8 = probs[pos]
                    # Hardware: prob.resize(16).asSInt -- zero-extend UINT8 to 16 bits,
                    # then reinterpret as SInt16 (value stays positive since bit 15 = 0)
                    prob_val = prob_uint8  # 0-255, zero-extended to 16 bits = positive SInt16
                    for dd in range(h_d):
                        v_val = to_sint(int(v_cached[dd]), 16)
                        product = to_sint(prob_val * v_val, 24)
                        acc[dd] = to_sint(acc[dd] + product, 24)

                # Store result: >> 8, saturate to INT8
                for dd in range(h_d):
                    attn_out[h * h_d + dd] = clamp_int8(acc[dd] >> 8)

            # --- O_PROJ via TDot (>> 8 -> INT8) ---
            o_out = hw_ternary_matvec(attn_out, weights[f"{prefix}.attn.o_proj"])
            step_debug[f"after_layer{layer_idx}_attn_out"] = o_out.tolist()

            # --- ATTN_RESIDUAL: x = clamp(x + attn_out, -128, 127) ---
            # Saturating add for better neural network behavior
            for i in range(d):
                s = to_sint(int(act[i]), 16) + to_sint(int(o_out[i]), 16)
                act[i] = clamp_int8(s)
            step_debug[f"after_layer{layer_idx}_attn_residual"] = act.tolist()

            # --- FF_NORM: RMSNorm ---
            gamma = weights[f"layer{layer_idx}_ff_norm"]
            normed = hw_rmsnorm(act, gamma)
            step_debug[f"after_layer{layer_idx}_ff_norm"] = normed.tolist()

            # --- FFN UP via TDot (64 -> 256, >> 8 -> INT8) ---
            up = hw_ternary_matvec(normed, weights[f"{prefix}.ff.up"])

            # --- RELU ---
            for i in range(len(up)):
                if up[i] < 0:
                    up[i] = 0
            step_debug[f"after_layer{layer_idx}_ff_up_relu"] = up.tolist()

            # --- FFN DOWN via TDot with multi-pass accumulation (256 -> 64) ---
            down = hw_ternary_matvec_accum(
                up, weights[f"{prefix}.ff.down"], d_in_slice=d
            )
            step_debug[f"after_layer{layer_idx}_ff_down"] = down.tolist()

            # --- FF_RESIDUAL: x = clamp(x + ff_out, -128, 127) ---
            for i in range(d):
                s = to_sint(int(act[i]), 16) + to_sint(int(down[i]), 16)
                act[i] = clamp_int8(s)
            step_debug[f"after_layer{layer_idx}_ff_residual"] = act.tolist()

        # ===========================================================
        # FINAL_NORM: RMSNorm with final_norm gamma
        # ===========================================================
        gamma = weights["final_norm"]
        normed_final = hw_rmsnorm(act, gamma)

        step_debug["after_final_norm"] = normed_final.tolist()

        # ===========================================================
        # OUTPUT_LOGITS: dot(query_vec, tok_emb[v]) for each vocab v
        # Matches Embedding.scala logit mode exactly:
        #   logitAcc += queryVec[i].resize(24) * embVal[i].resize(24)
        #   accumulated in INT24 with truncation
        # ===========================================================
        max_logit = -(2**23 - 1)  # -8388607, matches S(-8388607, 24 bits)
        max_token = 0
        all_logits = []

        for v in range(config.vocab_size):
            logit_acc = 0
            for i in range(d):
                q_val = to_sint(int(normed_final[i]), 24)
                e_val = to_sint(int(weights["tok_emb"][v, i]), 24)
                product = q_val * e_val
                logit_acc = to_sint(logit_acc + product, 24)

            all_logits.append(logit_acc)
            if logit_acc > max_logit:
                max_logit = logit_acc
                max_token = v

        step_debug["all_logits"] = all_logits

        next_token = max_token

        # ===========================================================
        # TEMPERATURE SAMPLING (matches SamplingUnit.scala exactly)
        # ===========================================================
        if inv_temp > 0:
            # Logit scale compensation: INT24 logits are ~4096x larger than float,
            # due to INT8 query × INT16 Q5.10 emb accumulation without d_model^-0.5.
            # Shift down to bring differences into a useful range for probability computation.
            LOGIT_SHIFT = 12

            # Pass 1: compute probabilities and total
            total_prob = 0
            for v_idx in range(config.vocab_size):
                diff = to_sint(all_logits[v_idx] - max_logit, 32)
                diff_scaled = to_sint(diff >> LOGIT_SHIFT, 32)
                shifted = to_sint((diff_scaled * inv_temp) >> 8, 32)
                p = max(0, min(256, 256 + shifted))
                total_prob += p
            total_prob = to_uint(total_prob, 24)

            # Advance LFSR
            lfsr_state = lfsr_step(lfsr_state)

            # Compute threshold: (lfsr[15:0] * total_prob) >> 16
            lfsr_low = lfsr_state & 0xFFFF
            threshold = to_uint((lfsr_low * total_prob) >> 16, 24)

            # Pass 2: cumulative sum sampling (recompute probs)
            cum_sum = 0
            for v_idx in range(config.vocab_size):
                diff = to_sint(all_logits[v_idx] - max_logit, 32)
                diff_scaled = to_sint(diff >> LOGIT_SHIFT, 32)
                shifted = to_sint((diff_scaled * inv_temp) >> 8, 32)
                p = max(0, min(256, 256 + shifted))
                cum_sum = to_uint(cum_sum + p, 24)
                if cum_sum > threshold:
                    next_token = v_idx
                    break
            else:
                next_token = 127  # fallback

            step_debug["sampling"] = {
                "inv_temp": inv_temp,
                "total_prob": total_prob,
                "lfsr_after": lfsr_state,
                "threshold": threshold,
                "selected": next_token,
            }

        step_debug["next_token"] = next_token
        step_debug["max_logit"] = max_logit
        debug_data.append(step_debug)

        # Only record generated tokens (after prompt processing)
        if step >= len(tokens) - 1:
            all_tokens.append(next_token)

    # Return only the generated tokens (predictions after each prompt token + new tokens)
    generated_tokens = [d["next_token"] for d in debug_data]
    return generated_tokens, debug_data


def generate_test_vectors(export_dir: str, output_dir: str, num_tokens: int = 16,
                          temperature: float = 0.0, seed: int = 0xDEADBEEF):
    """Generate golden test vectors for RTL validation."""
    os.makedirs(output_dir, exist_ok=True)

    weights, config = load_weights(export_dir)

    # Test prompt
    prompt = [ord(c) for c in "ROMEO:"]
    generated, debug = reference_inference(
        prompt, weights, config, max_new_tokens=num_tokens,
        temperature=temperature, seed=seed,
    )

    # Write test vectors
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(output_dir, "test_vectors.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "generated": generated,
                "debug": debug,
            },
            f,
            indent=2,
            cls=NumpyEncoder,
        )

    print(f"Prompt: {''.join(chr(t) for t in prompt)}")
    print(f"Generated {len(generated)} tokens: {generated}")
    gen_str = ""
    for t in generated:
        if 32 <= t < 127:
            gen_str += chr(t)
        else:
            gen_str += f"\\x{t:02x}"
    print(f"As text: \"{gen_str}\"")
    return generated, debug


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=str, help="Path to export directory")
    parser.add_argument("--output", type=str, default="test_vectors")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy argmax)")
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0xDEADBEEF,
                        help="LFSR seed for temperature sampling (hex-capable, e.g. 0xDEADBEEF)")
    parser.add_argument("--dump-step", type=int, default=None,
                        help="Print full debug dict for step N as JSON")
    args = parser.parse_args()

    generated, debug = generate_test_vectors(
        args.export_dir, args.output, args.tokens,
        temperature=args.temperature, seed=args.seed,
    )

    if args.dump_step is not None:
        step = args.dump_step
        if step < len(debug):
            print(f"\n=== Debug dump for step {step} ===")
            print(json.dumps(debug[step], indent=2, cls=type(
                "NE", (json.JSONEncoder,),
                {"default": lambda self, o: int(o) if isinstance(o, np.integer) else
                 float(o) if isinstance(o, np.floating) else
                 o.tolist() if isinstance(o, np.ndarray) else
                 json.JSONEncoder.default(self, o)}
            )))
        else:
            print(f"Error: step {step} out of range (0-{len(debug)-1})")
