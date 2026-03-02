"""ZyboGPT transformer model with BitLinear ternary layers.

Architecture:
- Token embedding (full precision, vocab=128, d_model=64)
- Learned positional embedding (full precision, ctx_len=128, d_model=64)
- 2x Decoder layers:
    - RMSNorm + Multi-head attention (2 heads, d_head=32) with ternary QKV+O projections
    - RMSNorm + FFN (d_ff=256) with ternary up/down projections, ReLU activation
- Final RMSNorm + output head (tied with embedding)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import (BitLinear, ste_round, ste_clamp_int8, ste_shift_right,
                         ste_shift_right_identity,
                         ste_round_clamp_int8, ste_round_clamp,
                         ste_truncate_int8, ste_hw_int8)
from .config import ZyboGPTConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, hw_mode: bool = False):
        super().__init__()
        self.eps = eps
        self.hw_mode = hw_mode
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hw_mode:
            return self._forward_hw(x)
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x.float() / rms
        return (x_norm * self.weight).to(x.dtype)

    def _forward_hw(self, x: torch.Tensor) -> torch.Tensor:
        """Hardware-accurate RMSNorm: LUT-based inv_sqrt, Q5.10 gamma, shifts.

        Matches hardware: y[i] = clamp((x[i] * invRms >> 8) * gamma_q >> 10, -128, 127)
        inv_rms stays in the gradient graph so normalization is self-regulating
        (if x grows, inv_rms shrinks, bounding the gradient naturally).
        """
        # x is in INT8 range [-128, 127]
        x_int = ste_round_clamp_int8(x)

        # Compute inv_rms from x (kept in gradient graph for self-regulation)
        sum_sq = (x_int * x_int).sum(dim=-1, keepdim=True)
        mean_sq = ste_shift_right(sum_sq, 6)
        lut_idx = ste_shift_right(mean_sq, 6).clamp(0, 255)
        bin_center = (lut_idx * 64 + 32).clamp(min=1)
        inv_rms = ste_round_clamp(16384.0 / torch.sqrt(bin_center.float()), 0, 16383)

        # Gamma quantized to Q5.10, clamped to prevent RMSNorm output saturation.
        # Output ≈ x * gamma * 64 / rms; gamma > 2.0 causes most values to hit ±128.
        weight_clamped = self.weight + (self.weight.clamp(-2.0, 2.0) - self.weight).detach()
        gamma_q = ste_round_clamp(weight_clamped * 1024, -32768, 32767)

        # prod1 = x * inv_rms >> 8
        prod1_trunc = ste_shift_right(x_int * inv_rms, 8)

        # prod2 = prod1_trunc * gamma_q >> 10
        y = ste_shift_right(prod1_trunc * gamma_q, 10)

        # Clamp to INT8
        y = ste_round_clamp_int8(y)
        return y


class Attention(nn.Module):
    """Multi-head attention with ternary QKV and O projections."""

    def __init__(self, config: ZyboGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.hw_mode = config.hw_mode

        self.q_proj = BitLinear(config.d_model, config.d_model, hw_mode=config.hw_mode)
        self.k_proj = BitLinear(config.d_model, config.d_model, hw_mode=config.hw_mode)
        self.v_proj = BitLinear(config.d_model, config.d_model, hw_mode=config.hw_mode)
        self.o_proj = BitLinear(config.d_model, config.d_model, hw_mode=config.hw_mode)

        # Cache for causal mask
        self._causal_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # KV cache for inference
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        if self.hw_mode:
            out = self._attention_hw(q, k, v, mask, T)
        else:
            # Fused scaled dot-product attention
            if mask is True:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif mask is not None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return out, new_kv

    def _get_causal_mask(self, T_q, T_kv, device):
        """Get or create cached causal mask."""
        key = (T_q, T_kv, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = torch.triu(
                torch.ones(T_q, T_kv, device=device, dtype=torch.bool),
                diagonal=T_kv - T_q + 1
            )
        return self._causal_mask_cache[key]

    def _hw_softmax(self, scores):
        """Hardware-accurate softmax: PL-exp approximation + normalization + UINT8.

        Matches Softmax.scala's 4-segment piecewise-linear exp:
          x < -24:        0
          -24 <= x < -8:  x + 24
          -8 <= x < -3:   64 + (x + 3) * 11
          -3 <= x <= 0:   256 + x * 64
        """
        max_score = scores.max(dim=-1, keepdim=True).values
        shifted = scores - max_score  # always <= 0

        # PL-exp (4 segments matching Softmax.scala)
        exp_approx = torch.zeros_like(shifted)
        exp_approx = torch.where((shifted >= -24) & (shifted < -8),
                                  shifted + 24, exp_approx)
        exp_approx = torch.where((shifted >= -8) & (shifted < -3),
                                  64 + (shifted + 3) * 11, exp_approx)
        exp_approx = torch.where(shifted >= -3,
                                  256 + shifted * 64, exp_approx)

        # Normalize (float division for differentiability)
        sum_exp = exp_approx.sum(dim=-1, keepdim=True).clamp(min=1)
        probs = exp_approx / sum_exp * 255.0
        probs_uint8 = ste_round_clamp(probs, 0, 255)
        return probs_uint8

    def _attention_hw(self, q, k, v, mask, T):
        """Hardware-accurate attention: integer scores, quantized softmax."""
        B, H, T_q, D = q.shape
        T_kv = k.shape[2]

        # Scores: q @ k.T (INT8 x INT8 -> ~INT24), then *45 >> 8 (hw 1/sqrt(32))
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T_q, T_kv)
        scores = ste_shift_right(scores * 45, 8)

        # Causal mask
        if mask is True:
            causal = self._get_causal_mask(T_q, T_kv, q.device)
            scores = scores.masked_fill(causal, -32767)
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, -32767)

        # Hardware-accurate PL-exp softmax (matches Softmax.scala)
        probs_uint8 = self._hw_softmax(scores)

        # Value: probs_uint8 @ v (UINT8 * INT8 -> accumulate), then >> 8, hw INT8
        out = torch.matmul(probs_uint8, v.float())
        out = ste_shift_right(out, 8)
        out = ste_hw_int8(out)
        return out.to(q.dtype)


class FeedForward(nn.Module):
    """FFN with ternary up/down projections and ReLU."""

    def __init__(self, config: ZyboGPTConfig):
        super().__init__()
        self.hw_mode = config.hw_mode
        self.up = BitLinear(config.d_model, config.d_ff, hw_mode=config.hw_mode)
        self.down = BitLinear(config.d_ff, config.d_model, hw_mode=config.hw_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = F.relu(h)
        if self.hw_mode:
            h = ste_round_clamp_int8(h)
        return self.down(h)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: ZyboGPTConfig):
        super().__init__()
        self.hw_mode = config.hw_mode
        self.attn_norm = RMSNorm(config.d_model, hw_mode=config.hw_mode)
        self.attn = Attention(config)
        self.ff_norm = RMSNorm(config.d_model, hw_mode=config.hw_mode)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention with residual (hw: .resize(8 bits) = truncation)
        h = self.attn_norm(x)
        attn_out, new_kv = self.attn(h, mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        if self.hw_mode:
            x = ste_hw_int8(x)

        # Pre-norm FFN with residual (hw: .resize(8 bits) = truncation)
        h = self.ff_norm(x)
        x = x + self.ff(h)
        if self.hw_mode:
            x = ste_hw_int8(x)

        return x, new_kv


class ZyboGPT(nn.Module):
    """Tiny ternary-quantized transformer for FPGA deployment."""

    def __init__(self, config: ZyboGPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.ctx_len, config.d_model)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, hw_mode=config.hw_mode)

        # Tie output head to embedding weights
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p is not self.tok_emb.weight:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = tokens.shape
        positions = torch.arange(start_pos, start_pos + T, device=tokens.device)

        if self.config.hw_mode:
            x = self._embed_hw(tokens, positions)
        else:
            x = self.tok_emb(tokens) + self.pos_emb(positions)

        # Causal mask: use is_causal=True for training/prefill (no explicit mask needed),
        # only build explicit mask when using KV cache with T>1 (rare)
        mask = None
        if T > 1 and kv_caches is not None and kv_caches[0][0].shape[2] > 0:
            cache_len = kv_caches[0][0].shape[2]
            mask = torch.tril(torch.ones(T, T, device=tokens.device)).unsqueeze(0).unsqueeze(0)
            ones = torch.ones(B, 1, T, cache_len, device=tokens.device)
            mask = torch.cat([ones, mask], dim=-1)
        elif T > 1:
            mask = True  # sentinel: tells Attention to use is_causal=True

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = layer(x, mask=mask, kv_cache=kv)
            new_kv_caches.append(new_kv)

        x = self.final_norm(x)

        if self.config.hw_mode:
            logits = self._logits_hw(x)
        else:
            logits = self.output_head(x) * (self.config.d_model ** -0.5)

        return logits, new_kv_caches

    def _embed_hw(self, tokens, positions):
        """Hardware-accurate embedding: Q5.10 tok+pos, >> 3, truncate to INT8.

        Matches hardware: (emb_sum >> 3).resize(8 bits) — truncation, not clamping.
        """
        tok_raw = self.tok_emb(tokens)   # float
        pos_raw = self.pos_emb(positions)  # float

        # Quantize to Q5.10 (multiply by 1024, round, clamp INT16)
        tok_q = ste_round_clamp(tok_raw * 1024, -32768, 32767)
        pos_q = ste_round_clamp(pos_raw * 1024, -32768, 32767)

        # Sum and shift >> 3, hw INT8 (truncate in Phase 2, clamp in Phase 1)
        x = ste_shift_right(tok_q + pos_q, 3)
        x = ste_hw_int8(x)
        return x

    def _logits_hw(self, x):
        """Hardware-accurate logits: INT8 x Q5.10 embedding, scaled for cross-entropy.

        Hardware uses argmax on raw INT24 (scale-invariant), but training needs
        well-scaled logits. x is INT8 (contains ~64x RMSNorm scale-up), emb_q is
        Q5.10 (1024x). Undo both to get float-scale logits, then apply 1/sqrt(d_model).
        """
        # x is INT8 range from final_norm
        emb_q = ste_round_clamp(self.tok_emb.weight * 1024, -32768, 32767)  # Q5.10
        # Raw INT24 dot product (hardware uses argmax on this)
        logits_raw = torch.matmul(x, emb_q.T)
        # Undo 64x RMSNorm scale-up and 1024x Q5.10, plus 1/sqrt(d_model)
        logits = logits_raw / (64.0 * 1024.0 * math.sqrt(self.config.d_model))
        return logits

    def count_params(self) -> dict[str, int]:
        """Count ternary vs full-precision parameters."""
        ternary = 0
        full = 0
        for name, p in self.named_parameters():
            if "bitlinear" in name.lower() or any(
                k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "up", "down"]
            ):
                if "weight" in name and p.dim() == 2:
                    ternary += p.numel()
                    continue
            full += p.numel()
        return {"ternary": ternary, "full_precision": full, "total": ternary + full}

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[int]:
        """Autoregressive generation with KV cache."""
        self.eval()
        tokens = prompt_tokens.unsqueeze(0) if prompt_tokens.dim() == 1 else prompt_tokens
        B, T = tokens.shape
        generated = tokens[0].tolist()

        # Prefill
        logits, kv_caches = self.forward(tokens)
        pos = T

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            if temperature == 0:
                next_token = torch.argmax(probs, dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated.append(next_token.item())
            next_input = next_token.unsqueeze(-1)  # (1,) -> (1, 1)

            logits, kv_caches = self.forward(next_input, kv_caches=kv_caches, start_pos=pos)
            pos += 1

            if pos >= self.config.ctx_len:
                break

        return generated
