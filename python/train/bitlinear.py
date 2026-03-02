"""BitLinear layer: ternary weights with STE, INT8 activations.

Implements the core quantization scheme:
- Weights: {-1, 0, +1} via round-to-nearest with straight-through estimator
- Activations: INT8 with per-tensor absmax scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# STE (Straight-Through Estimator) helpers for hw_mode
# Fused custom autograd functions for minimal intermediate tensors.
# ================================================================

class _STERoundClampInt8(torch.autograd.Function):
    """Fused round + clamp[-128,127] with identity gradient."""
    @staticmethod
    def forward(ctx, x):
        return x.round().clamp(-128, 127)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _STETruncateInt8(torch.autograd.Function):
    """Truncate to signed INT8 (lower 8 bits, wrap around) with identity gradient.

    Matches hardware .resize(8 bits): takes lower 8 bits with sign extension.
    E.g. 187 -> -69, -200 -> 56.
    """
    @staticmethod
    def forward(ctx, x):
        return ((x.round() + 128) % 256) - 128

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _STEShiftRight(torch.autograd.Function):
    """Floor(x / 2^n) with 1/2^n gradient."""
    @staticmethod
    def forward(ctx, x, n):
        ctx.n = n
        return (x / (1 << n)).floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / (1 << ctx.n), None


class _STEShiftRightIdentity(torch.autograd.Function):
    """Floor(x / 2^n) with identity gradient (no attenuation)."""
    @staticmethod
    def forward(ctx, x, n):
        return (x / (1 << n)).floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _STERoundClamp(torch.autograd.Function):
    """Fused round + clamp with identity gradient."""
    @staticmethod
    def forward(ctx, x, lo, hi):
        return x.round().clamp(lo, hi)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def ste_round_clamp_int8(x: torch.Tensor) -> torch.Tensor:
    """Round and clamp to [-128, 127] with identity gradient (fused)."""
    return _STERoundClampInt8.apply(x)


def ste_truncate_int8(x: torch.Tensor) -> torch.Tensor:
    """Truncate to signed INT8 (wrap around) with identity gradient.

    Matches hardware .resize(8 bits): takes lower 8 bits with sign extension.
    """
    return _STETruncateInt8.apply(x)


# ================================================================
# Toggleable hw INT8 mode: clamp (Phase 1) vs truncate (Phase 2)
# ================================================================
_hw_truncation_enabled = False


def set_hw_truncation(enabled: bool):
    """Toggle between clamping and truncation for hw_mode INT8 outputs.

    Phase 1 (enabled=False): clamp — stable gradients, fast convergence.
    Phase 2 (enabled=True):  truncate — matches hardware .resize(8 bits).
    """
    global _hw_truncation_enabled
    _hw_truncation_enabled = enabled


def ste_hw_int8(x: torch.Tensor) -> torch.Tensor:
    """INT8 quantization for hardware datapath outputs.

    Uses truncation when _hw_truncation_enabled (matches hw .resize(8 bits)),
    clamping otherwise (stable for training).
    """
    if _hw_truncation_enabled:
        return _STETruncateInt8.apply(x)
    return _STERoundClampInt8.apply(x)


def ste_shift_right(x: torch.Tensor, n: int) -> torch.Tensor:
    """Floor(x / 2^n) with 1/2^n gradient (natural gradient of the division)."""
    return _STEShiftRight.apply(x, n)


def ste_shift_right_identity(x: torch.Tensor, n: int) -> torch.Tensor:
    """Floor(x / 2^n) with identity gradient."""
    return _STEShiftRightIdentity.apply(x, n)


def ste_round_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Round and clamp to [lo, hi] with identity gradient (fused)."""
    return _STERoundClamp.apply(x, lo, hi)


# Keep simple wrappers for backward compat
def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with identity gradient (straight-through estimator)."""
    return x + (x.round() - x).detach()


def ste_clamp_int8(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [-128, 127] with identity gradient."""
    return x + (x.clamp(-128, 127) - x).detach()


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean threshold.

    Returns (ternary_weights, scale) where scale = mean(|w|).
    """
    scale = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / scale
    # Round to nearest ternary value
    w_ternary = w_scaled.round().clamp(-1, 1)
    return w_ternary, scale


def int8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to INT8 with per-tensor absmax scaling.

    Returns (quantized_int8, scale) where x ≈ quantized_int8 * scale.
    """
    absmax = x.abs().max().clamp(min=1e-5)
    scale = absmax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127)
    return x_int8, scale


class BitLinear(nn.Module):
    """Linear layer with ternary weights and INT8 activations.

    During forward pass:
    1. Quantize weights to ternary using STE
    2. Quantize input activations to INT8
    3. Compute y = x_q @ w_q.T (integer arithmetic)
    4. Rescale: y_float = y * x_scale * w_scale

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 hw_mode: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hw_mode = hw_mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Kaiming init
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hw_mode:
            return self._forward_hw(x)

        # Ternary weight quantization with STE
        w_ternary, w_scale = ternary_quantize(self.weight)
        # Straight-through estimator: use ternary in forward, full precision in backward
        w_q = self.weight + (w_ternary * w_scale - self.weight).detach()

        # INT8 activation quantization with STE
        x_int8, x_scale = int8_quantize(x)
        x_q = x + (x_int8 * x_scale - x).detach()

        y = F.linear(x_q, w_q, self.bias)
        return y

    def _forward_hw(self, x: torch.Tensor) -> torch.Tensor:
        """Hardware-accurate forward: ternary matmul >> 4, hw INT8.

        Matches hardware: (acc >> 4).resize(8 bits) — truncation in Phase 2.
        """
        # Ternary weights with STE (no w_scale!)
        w_ternary, _ = ternary_quantize(self.weight)
        w_q = self.weight + (w_ternary - self.weight).detach()

        # Ensure input is INT8 range (fused round+clamp — input is already in range)
        x_q = ste_round_clamp_int8(x)

        # Integer matmul, >> 4, hw INT8 (truncate in Phase 2, clamp in Phase 1)
        y = F.linear(x_q, w_q)
        y = ste_shift_right(y, 4)
        y = ste_hw_int8(y)
        return y

    def get_ternary_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the actual ternary weights for export (no STE)."""
        with torch.no_grad():
            return ternary_quantize(self.weight)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"
