"""ZyboGPT model configuration."""

from dataclasses import dataclass


@dataclass
class ZyboGPTConfig:
    vocab_size: int = 128         # ASCII
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    ctx_len: int = 128
    head_dim: int = 32            # d_model // n_heads

    # Training
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_steps: int = 50_000
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Quantization
    activation_bits: int = 8      # INT8 activations
    activation_scale_bits: int = 16  # FP16 scales during training

    # Hardware-accurate training mode
    hw_mode: bool = False         # When True, forward pass simulates FPGA integer arithmetic

    # Export
    ternary_pack_bits: int = 8    # 5 ternary trits packed per byte (3^5=243<256)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
