# ZyboGPT SpinalHDL Hardware

FPGA accelerator for ternary-quantized transformer inference on the Zybo Z7-10 (Zynq xc7z010clg400-1).

## Overview

The accelerator performs single-token autoregressive inference for a 2-layer ternary transformer at 150 MHz. All model weights are stored in BRAM and decoded at runtime. Compute is time-multiplexed across layers using a single TDot (ternary dot product) unit and 8 shared INT8 MACs.

### Measured Performance

| Metric | Value |
|--------|-------|
| Clock | 150 MHz |
| Throughput (on-hardware BENCH) | 3,072 tok/s |
| Throughput (simulation) | 5,441 tok/s |
| Avg cycles/token (sim, 16 pos) | 27,439 |
| Cycles/token (pos 0, sim) | 24,987 |
| Board power | ~2-3 W |

### Resource Utilization (xc7z010clg400-1, placed)

| Resource | Used | Available | Util% |
|----------|------|-----------|-------|
| Slice LUTs | 14,952 | 17,600 | 85% |
| Slice Registers | 13,317 | 35,200 | 38% |
| Block RAM Tiles | 30.5 | 60 | 51% |
| DSP48E1 | 67 | 80 | 84% |
| Slices | 4,400 | 4,400 | 100% |

### Per-Module Breakdown (synthesis)

| Module | LUTs | DSPs | BRAM |
|--------|------|------|------|
| Attention | 4,524 | 32 | 1 RAMB18 |
| FeedForward | 4,781 | 0 | 1 RAMB18 |
| WeightBram (+ TDot + decoders) | 2,668 | 20 | 9 RAMB36 |
| Embedding | 1,184 | 0 | 8 RAMB36 + 3 RAMB18 |
| Sequencer (+ SamplingUnit) | 1,087 | 2 | 1 RAMB18 |
| RMSNorm (attn, x2) | 219 + 403 | 2 + 2 | 1 + 1 RAMB18 |
| Softmax | 168 | 1 | 3 RAMB18 |
| KvCache | 113 | 0 | 8 RAMB36 |
| Int8MacArray | 10 | 8 | 0 |
| AxiLiteSlave | 96 | 0 | 0 |
| TransformerLayer (glue) | 144 | 0 | 0 |

## Prerequisites

- **JDK** 11+ (for sbt/Scala)
- **sbt** 1.x (Scala Build Tool)
- **Verilator** (for SpinalHDL simulations)

Dependencies are managed by sbt (SpinalHDL 1.10.2a, Scala 2.13.12, ScalaTest 3.2.17).

## Usage

```bash
# From project root

# Generate Verilog
make spinal
# Output: hw/gen/ZyboGPTTop.v

# Run all simulation tests
make spinal-test
```

Or using sbt directly:

```bash
cd hw

# Generate Verilog
sbt "runMain zybogpt.ZyboGPTVerilog"

# Run individual tests
sbt "Test/runMain zybogpt.TDotUnitSim"
sbt "Test/runMain zybogpt.RMSNormSim"
sbt "Test/runMain zybogpt.ZyboGPTSim"

# Run pipeline debug (stage-by-stage HW vs Python comparison)
sbt "Test/runMain zybogpt.ZyboGPTPipelineDebugSim"
```

**Note:** Weight `.mem` files must exist in `export/` before Verilog generation. Run `make export` first.

## Architecture

```
+-------------------------------------------------------------------+
|  Zynq PS (ARM Cortex-A9)                                          |
|           | AXI4-Lite GP0 (0x43C0_0000)                           |
+-----------v-------------------------------------------------------+
|  PL (Programmable Logic) @ 150 MHz                                |
|                                                                   |
|  AxiLiteSlave --> Sequencer FSM                                   |
|                     |                                             |
|                     +--> Embedding (tok + pos BRAM)               |
|                     |                                             |
|                     +--> TransformerLayer (x2 time-multiplexed)   |
|                     |      +-- RMSNorm --> Attention              |
|                     |      |                +-- Q/K/V proj (TDot) |
|                     |      |                +-- QK^T (8 MACs)     |
|                     |      |                +-- Softmax           |
|                     |      |                +-- Attn*V (8 MACs)   |
|                     |      |                +-- O proj (TDot)     |
|                     |      +-- RMSNorm --> FeedForward            |
|                     |                       +-- Up proj (TDot)    |
|                     |                       +-- ReLU              |
|                     |                       +-- Down proj (TDot)  |
|                     |                                             |
|                     +--> Final RMSNorm                            |
|                     +--> Logit (tied embedding, 8x parallel)      |
|                     +--> Argmax / SamplingUnit                    |
|                                                                   |
|  Shared Resources:                                                |
|    1 TDotUnit <-- WeightBram (serial 32-row loading)              |
|    8 INT8 MACs <-- KV Cache BRAM                                  |
+-------------------------------------------------------------------+
```

### Inference Pipeline (Sequencer FSM)

```
IDLE -> EMBED -> EMB_READ -> LAYER_LOOP -> FINAL_NORM -> OUTPUT_LOGITS -> ARGMAX -> DONE
                              (2 layers)                            +-> SAMPLING -+
```

When `inv_temp == 0` (SAMPLING register), greedy argmax is used. When `inv_temp > 0`, the SamplingUnit performs two-pass temperature sampling with a Galois LFSR for deterministic randomness.

Each layer executes: Attn RMSNorm -> Attention -> Residual -> FF RMSNorm -> FFN -> Residual

### TDot Weight Loading (ZyboGPT.scala FSM)

The TDot controller transparently handles multi-cycle weight preloading:

```
IDLE --(tdotStart)--> LOADING --(loadDone)--> COMPUTE (32 rows serial) --> DONE
```

1. TransformerLayer requests a TDot operation (sets weight address + start)
2. WeightBram reads packed trits from BRAM via 4 decoders (20 trits/cycle)
3. For each of the 32 output rows: ASSEMBLE (4 reads) -> FIRE (1 cycle) -> CAPTURE
4. TDotUnit computes one 64-wide ternary dot product per FIRE
5. Results accumulated and returned to TransformerLayer

## Module Reference

### Top Level

| Module | File | Description |
|--------|------|-------------|
| `ZyboGPTTop` | `ZyboGPT.scala` | Top-level integration, TDot controller FSM, norm gamma loading |
| `ZyboGPTVerilog` | `ZyboGPT.scala` | Verilog generation entry point |

### Control

| Module | File | Description |
|--------|------|-------------|
| `AxiLiteSlave` | `AxiLiteSlave.scala` | AXI4-Lite register interface to Zynq PS |
| `Sequencer` | `Sequencer.scala` | Top-level inference FSM (IDLE through DONE) |

### Compute Units

| Module | File | Description |
|--------|------|-------------|
| `TDotUnit` | `TDotUnit.scala` | 64-wide ternary dot product (mux + 3-stage pipelined adder tree) |
| `Int8MacUnit` | `Int8MacUnit.scala` | INT8 multiply-accumulate with pipeline registers |
| `Int8MacArray` | `Int8MacArray.scala` | 8 parallel MAC lanes for attention score and value computation |
| `WeightDecoder` | `WeightDecoder.scala` | Unpack 5 trits from 1 byte (base-3 decode via r*171 multiply) |

### Normalization and Nonlinearities

| Module | File | Description |
|--------|------|-------------|
| `RMSNorm` | `RMSNorm.scala` | Integer RMSNorm with 256-entry inv_sqrt LUT |
| `Softmax` | `Softmax.scala` | Piecewise-linear exp LUT + reciprocal LUT |
| `SamplingUnit` | `SamplingUnit.scala` | Temperature sampling (2-pass + Galois LFSR) |

### Memory

| Module | File | Description |
|--------|------|-------------|
| `Embedding` | `Embedding.scala` | Token + positional embedding BRAM, 8x parallel logit computation |
| `WeightBram` | `WeightBram.scala` | Packed ternary weight storage + serial TDotUnit + 4 decoders |
| `KvCache` | `KvCache.scala` | Per-layer, per-head K/V circular buffer (bit-shift addressing) |

### Transformer

| Module | File | Description |
|--------|------|-------------|
| `Attention` | `Attention.scala` | Multi-head attention FSM (Q/K/V -> scores -> softmax -> AV -> O proj) |
| `FeedForward` | `FeedForward.scala` | Up projection -> ReLU -> Down projection (compile-time unrolled) |
| `TransformerLayer` | `TransformerLayer.scala` | Pre-norm Attention + FFN with residual connections |

### Configuration

| Module | File | Description |
|--------|------|-------------|
| `ZyboGPTHwConfig` | `Config.scala` | All hardware parameters (dimensions, array sizes, clock) |
| `SatInt8` | `Config.scala` | Saturating INT8 clamp helper |
| `WeightInit` | `WeightInit.scala` | Generated constants: BRAM offsets, scale values, norm gammas |

## AXI4-Lite Register Map

Base address: `0x43C0_0000` (Zynq AXI GP0)

| Offset | Name | Access | Bits | Description |
|--------|------|--------|------|-------------|
| 0x00 | CONTROL | W | [0] start, [1] reset, [3:2] mode | Control register |
| 0x04 | STATUS | R | [0] busy, [1] done, [31:16] cycle_count | Status register |
| 0x08 | TOKEN_IN | W | [6:0] token ID | Input token (0-127 ASCII) |
| 0x0C | TOKEN_OUT | R | [6:0] token ID | Output token (argmax result) |
| 0x10 | POSITION | W | [6:0] position | Sequence position (0-127) |
| 0x14 | CYCLE_LO | R | [31:0] | Cycle counter (lower 32 bits) |
| 0x18 | CYCLE_HI | R | [31:0] | Cycle counter (upper 32 bits) |
| 0x1C | CONFIG | R | [7:0] d_model, [15:8] n_layers, [23:16] ctx_len, [31:24] vocab | Model config |
| 0x20 | SAMPLING | W/R | [15:0] inv_temp | Temperature sampling (0 = greedy argmax) |
| 0x24 | SEED | W/R | [31:0] seed | LFSR seed for temperature sampling |

## Simulation Tests

All tests use SpinalHDL simulation via Verilator. Run with `make spinal-test`.

**Important:** Always clear the Verilator cache before running tests. The `make spinal-test` target does this automatically. Do not run sbt directly without first running `rm -rf hw/simWorkspace/`.

### Unit Tests

| Test | File | What it verifies |
|------|------|------------------|
| `TDotUnitSim` | `TDotUnitSim.scala` | Ternary dot product correctness (zero, all-+1, all--1, mixed, random) |
| `TDotArraySim` | `TDotArraySim.scala` | Parallel dot products with varied weight patterns |
| `Int8MacUnitSim` | `Int8MacSim.scala` | Single MAC: multiply, accumulate, clear, overflow |
| `Int8MacArraySim` | `Int8MacSim.scala` | 8-lane parallel dot products |
| `WeightDecoderSim` | `WeightDecoderSim.scala` | All 243 base-3 encodings decode correctly |
| `RMSNormSim` | `RMSNormSim.scala` | Against floating-point reference |
| `SoftmaxSim` | `SoftmaxSim.scala` | Against floating-point reference |
| `KvCacheSim` | `KvCacheSim.scala` | Write/read across layers, heads, positions |
| `AxiLiteSim` | `AxiLiteSim.scala` | Register read/write, start/busy/done handshake |
| `SamplingSim` | `SamplingSim.scala` | Temperature sampling: determinism, LFSR sequence, temperature sensitivity |
| `ZyboGPTSim` | `ZyboGPTSim.scala` | Full system AXI-driven inference |

### Integration Tests

| Test | File | What it verifies |
|------|------|------------------|
| `ZyboGPTRomeoSim` | `ZyboGPTRomeoSim.scala` | Full 16-token "ROMEO:" inference, bit-accurate match against Python reference (11/11 tokens) |
| `ZyboGPTPipelineDebugSim` | `ZyboGPTPipelineDebugSim.scala` | Per-stage activation comparison against Python reference at position 6 (16 pipeline stages, zero tolerance) |

Run the pipeline debug test with `make pipeline-debug`.

## Weight Initialization

Weights are loaded at SpinalHDL **elaboration time** (before Verilog generation), not at synthesis time:

- **Ternary weights**: `WeightBram` reads `../export/weights_ternary_32b.mem` via `scala.io.Source`, initializes BRAM with `Mem.init()`
- **Token/pos embeddings**: `Embedding` reads `../export/tok_emb_16b.mem` and `../export/pos_emb_16b.mem`
- **Wide token embedding**: `Embedding` reads `../export/tok_emb_16b.mem` and packs 8 INT16 values per 128-bit word for parallel logit computation
- **Scale values**: `WeightInit.scaleValues` array (12 INT16 values) compiled into `WeightBram`
- **Norm gammas**: `WeightInit.normGammas` array (320 INT16 values) loaded into `normGammaMem` in `ZyboGPTTop`

The `.mem` files must be present in `export/` before running `make spinal`.

## Key Design Decisions

- **Time-multiplexed layers**: Both transformer layers share the same TDot and MAC hardware
- **Single TDotUnit**: Replaced 32 parallel TDotUnits with serial row-by-row computation to fit in LUT budget
- **8 shared MACs**: Attention Q@K^T and attn@V use the same Int8MacArray, serialized over inner dimension
- **BRAM-backed buffers**: All large buffers use Mem() (BRAM) instead of Vec(Reg()) to avoid mux tree explosion
- **Address/data interfaces**: Vec outputs replaced with BRAM addr/data ports to eliminate read mux trees
- **Direct LUTs over iterative algorithms**: RMSNorm uses a 256-entry inv_sqrt LUT; Softmax uses piecewise-linear exp BRAM LUT
- **Compile-time unrolling**: FeedForward DOWN_PROJ and Attention head loops use constant indices to avoid runtime mux trees
- **Elaboration-time weight loading**: Avoids JVM 64KB static initializer limit that would occur with inline Scala arrays

## Directory Structure

```
hw/
  build.sbt                    # sbt build config (SpinalHDL 1.10.2a)
  src/main/scala/zybogpt/
    Config.scala               # Hardware parameters + SatInt8 helper
    ZyboGPT.scala              # Top-level + TDot controller + Verilog entry point
    Sequencer.scala            # Inference FSM
    AxiLiteSlave.scala         # AXI4-Lite register interface
    TransformerLayer.scala     # Layer orchestration (norm + attn + ffn)
    Attention.scala            # Multi-head attention FSM
    FeedForward.scala          # Up -> ReLU -> Down (compile-time unrolled)
    TDotUnit.scala             # Single ternary dot product (3-stage pipeline)
    Int8MacUnit.scala          # INT8 MAC with pipeline registers
    Int8MacArray.scala         # 8 parallel MACs
    WeightDecoder.scala        # 1.6-bit ternary unpacker
    WeightBram.scala           # Weight storage + serial TDot + decoders
    Embedding.scala            # Token + positional embeddings + parallel logit
    KvCache.scala              # KV cache BRAM (bit-shift addressing)
    RMSNorm.scala              # Integer RMSNorm
    Softmax.scala              # Integer softmax (BRAM LUTs)
    SamplingUnit.scala         # Temperature sampling (2-pass + Galois LFSR)
    WeightInit.scala           # Generated constants (scales, gammas, offsets)
    LUT_ANALYSIS.md            # Detailed LUT optimization analysis
  src/test/scala/zybogpt/
    TDotUnitSim.scala          # TDot unit tests
    Int8MacSim.scala           # MAC unit + array tests
    WeightDecoderSim.scala     # Decoder tests (all 243 values)
    RMSNormSim.scala           # Norm accuracy tests
    SoftmaxSim.scala           # Softmax accuracy tests
    KvCacheSim.scala           # Cache read/write tests
    AxiLiteSim.scala           # AXI register tests
    SamplingSim.scala          # Temperature sampling tests
    ZyboGPTSim.scala           # Full system test
    ZyboGPTRomeoSim.scala      # Integration test (16-token match vs Python)
    ZyboGPTPipelineDebugSim.scala  # Per-stage pipeline debug vs Python reference
  gen/                         # Generated Verilog output
  constraints/
    zybo_z7_10.xdc             # Timing and configuration constraints
```
