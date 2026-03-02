# ZyboGPT Python Training Pipeline

Train a ternary-quantized character-level transformer for deployment on Zybo Z7-10 FPGA.

## Overview

The training pipeline produces a tiny transformer with **ternary weights** ({-1, 0, +1}) and **INT8 activations**, then exports all parameters in hardware-ready formats for SpinalHDL elaboration and Vivado synthesis.

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Vocabulary | 128 (ASCII) |
| Embedding dim (`d_model`) | 64 |
| Attention heads | 2 |
| Decoder layers | 2 |
| FFN hidden dim (`d_ff`) | 256 |
| Context length | 128 |
| Head dim | 32 |
| Total params | ~123K (~115K ternary + ~8.4K full-precision) |

**Architecture details:**
- Pre-norm decoder-only transformer (GPT-style)
- RMSNorm (not LayerNorm)
- ReLU activation in FFN (not GELU/SiLU)
- Tied embeddings: output head shares weights with token embedding
- Causal (autoregressive) attention mask

### Quantization: BitLinear with STE

All linear layers use `BitLinear` from `bitlinear.py`:

- **Weights**: Quantized to {-1, 0, +1} using round-to-nearest with an absmean threshold. The straight-through estimator (STE) passes gradients through the quantization during backprop, enabling standard optimizer updates on full-precision shadow weights.
- **Activations**: Per-tensor INT8 quantization using absmax scaling. STE also applies here.
- **Scales**: Each ternary projection stores a per-layer scale factor (absmax of the FP32 weight matrix), exported as INT16 in Q5.10 fixed-point.

Only embeddings and RMSNorm gamma weights remain in full precision (INT16 Q5.10 after export).

## Prerequisites

```bash
# From project root
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA is recommended but not required.

## Usage

### Training (Two-Phase Curriculum)

Training uses a two-phase curriculum:

**Phase 1: Float pretrain** — ternary weight STE but float activations/norms:
```bash
make train-phase1
# Or directly:
venv/bin/python -m python.train.train --steps 50000 --lr 5e-4 --device auto --save-dir checkpoints/phase1
```

**Phase 2: HW-mode fine-tune** — INT8 activations matching FPGA datapath:
```bash
make train-phase2
# Or directly:
venv/bin/python -m python.train.train --steps 50000 --lr 3e-4 --hw-mode \
    --device auto --save-dir checkpoints/phase2 --resume checkpoints/phase1/best.pt
```

**Or run both phases:**
```bash
make train
```

**CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 50000 | Training steps |
| `--lr` | 5e-4 | Learning rate |
| `--device` | auto | `cuda`, `cpu`, or `auto` |
| `--save-dir` | checkpoints | Checkpoint output directory |
| `--hw-mode` | False | Enable hardware-accurate INT8 forward pass |
| `--resume` | None | Resume from checkpoint (model weights only, fresh optimizer) |
| `--continue-from` | None | Continue training (full state restore) |

**Training hyperparameters** (configured in `config.py`):
| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Batch size | 256 | 256 |
| Learning rate | 5e-4 | 3e-4 |
| Optimizer | AdamW (betas=0.9/0.95) | AdamW (betas=0.9/0.95) |
| Weight decay | 0.1 | 0.1 |
| LR schedule | Cosine annealing with 1K warmup | Cosine annealing with 1K warmup |
| Gradient clipping | 1.0 | 1.0 |
| hw_mode | False | True |

**Outputs:**
- `checkpoints/phase{1,2}/best.pt` — Lowest validation loss
- `checkpoints/phase{1,2}/final.pt` — Final checkpoint

**Expected results:**
- Phase 1: val_loss ~1.81
- Phase 2: val_loss ~2.23 (fundamental INT8+ternary capacity limit)

### Export

```bash
make export

# Or directly:
venv/bin/python scripts/generate_weights.py checkpoints/phase2/best.pt --output export
```

This runs the full export pipeline and copies `WeightInit.scala` into the hardware source tree.

**Generated files in `export/`:**

| File | Format | Description |
|------|--------|-------------|
| `weights_ternary.bin` | Raw bytes | Packed 1.6-bit ternary weights |
| `weights_ternary.mem` | Hex (1 byte/line) | Same data, one hex byte per line |
| `weights_ternary_32b.mem` | Hex (4 bytes/line) | 32-bit wide BRAM init for SpinalHDL |
| `weights_ternary_32b.coe` | Vivado COE | Vivado BRAM init format |
| `weights_full.bin` | Raw INT16 | Embeddings + norms as Q5.10 |
| `weights_full_16b.mem` | Hex (2 bytes/line) | Same data, hex format |
| `tok_emb_16b.mem` | Hex (2 bytes/line) | Token embedding (128 x 64 INT16) |
| `pos_emb_16b.mem` | Hex (2 bytes/line) | Positional embedding (128 x 64 INT16) |
| `WeightInit.scala` | Scala | BRAM offsets, scale values, norm gammas |
| `meta.json` | JSON | Metadata (shapes, scales, byte offsets) |

### Validation

```bash
make validate

# Or directly:
venv/bin/python scripts/validate_model.py checkpoints/phase2/best.pt --export export
```

Runs 5 validation steps:
1. PyTorch FP32 inference (baseline)
2. Export to FPGA format
3. INT8 bit-accurate reference inference
4. Token match comparison (FP32 vs INT8)
5. Generate RTL test vectors to `export/test_vectors/`

### Text Generation

```bash
# Float PyTorch inference:
venv/bin/python scripts/generate.py --prompt "ROMEO:" --temperature 0.5

# Bit-accurate INT8 reference inference (matches FPGA):
venv/bin/python -m python.train.reference_inference export/ --temperature 0.5 --tokens 16
```

## File Descriptions

### Core Training

| File | Description |
|------|-------------|
| `config.py` | `ZyboGPTConfig` dataclass with all model and training hyperparameters |
| `bitlinear.py` | `BitLinear` layer with ternary weight and INT8 activation quantization via STE |
| `model.py` | Full transformer: `RMSNorm`, `Attention`, `FeedForward`, `DecoderLayer`, `ZyboGPT` |
| `tokenizer.py` | `ASCIITokenizer` — direct ASCII char-to-int mapping (vocab size 128) |
| `dataset.py` | `ShakespeareDataset` — downloads Tiny Shakespeare, produces (input, target) pairs |
| `train.py` | Training loop with AdamW, cosine LR, gradient clipping, checkpointing |

### Export and Validation

| File | Description |
|------|-------------|
| `export.py` | `export_model()` — converts PyTorch checkpoint to all hardware formats |
| `reference_inference.py` | Bit-accurate INT8/ternary inference in NumPy for RTL validation |

## Ternary Weight Packing

Ternary weights use the TerEffic 1.6-bit packing scheme (5 trits per byte):

```
Ternary value:  -1  0  +1
Encoded as:      0  1   2   (unsigned)
```

Five trits are packed into one byte as a base-3 number:
```
byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
```

Maximum value: 2+6+18+54+162 = 242, fits in a byte.

## Q5.10 Fixed-Point Format

Full-precision values (embedding weights, norm gammas, scales) are exported as 16-bit signed integers in Q5.10 format:

```
value_real = value_int16 / 1024.0
Range: -32.0 to +31.999
```

## Pipeline Debugging

`reference_inference.py` captures intermediate activations at every pipeline stage boundary. Use `--dump-step N` to print the full debug dict for step N:

```bash
venv/bin/python -m python.train.reference_inference export/ --dump-step 6

# With temperature sampling (matches hardware SamplingUnit exactly):
venv/bin/python -m python.train.reference_inference export/ --temperature 0.5 --seed 0xDEADBEEF --tokens 16
```

**CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--tokens` | 16 | Number of tokens to generate |
| `--temperature` | 0.0 | Sampling temperature (0.0 = greedy argmax) |
| `--seed` | 0xDEADBEEF | LFSR seed for temperature sampling |
| `--dump-step` | None | Print full debug dict for step N as JSON |

These captures are used by the SpinalHDL `ZyboGPTPipelineDebugSim` testbench to verify hardware matches the Python reference at every pipeline stage with zero tolerance.

## Dataset

Tiny Shakespeare (~1.1 MB, ~1.1M characters) is automatically downloaded on first training run. Split 90/10 for training/validation. Each sample is a 129-character window: 128 input tokens predicting 128 next tokens.
