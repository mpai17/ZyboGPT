# LUT Utilization Analysis

## Summary

ZyboGPT fits on the xc7z010clg400-1 (Zybo Z7-10) at **85% LUT utilization**
after a 6-phase optimization that reduced synthesis from 128K LUTs to 15K LUTs.

## Current Utilization

### Implementation (placed)

| Resource | Used | Available | Util% |
|----------|------|-----------|-------|
| Slice LUTs | 14,952 | 17,600 | 84.95% |
| Slice Registers | 13,317 | 35,200 | 37.83% |
| Block RAM Tiles | 30.5 | 60 | 50.83% |
| RAMB36E1 | 25 | 60 | 41.67% |
| RAMB18E1 | 11 | 120 | 9.17% |
| DSP48E1 | 67 | 80 | 83.75% |
| CARRY4 | 2,483 | — | — |
| F7 Muxes | 387 | 8,800 | 4.40% |
| F8 Muxes | 144 | 4,400 | 3.27% |
| Slices | 4,400 | 4,400 | 100% |

### Synthesis — per-module breakdown (hierarchical)

| Module | LUTs | FFs | RAMB36 | RAMB18 | DSPs |
|--------|------|-----|--------|--------|------|
| **Attention** | 4,524 | 3,342 | 0 | 1 | 32 |
| **FeedForward** | 4,781 | 2,857 | 0 | 1 | 0 |
| **WeightBram** | 2,668 | 1,688 | 9 | 0 | 20 |
|   WeightBram (core) | 1,493 | 1,169 | 9 | 0 | 0 |
|   TDotUnit | 1,088 | 434 | 0 | 0 | 0 |
|   WeightDecoders (×4) | 87 | 85 | 0 | 0 | 20 |
| **TransformerLayer** (glue) | 144 | 1,551 | 0 | 0 | 0 |
| **Embedding** | 1,184 | 348 | 8 | 3 | 0 |
| **Sequencer** | 950 | 623 | 0 | 1 | 0 |
| **RMSNorm** (×2) | 219 + 403 | 612 + 612 | 0 | 1 + 1 | 2 + 2 |
| **Softmax** | 168 | 127 | 0 | 3 | 1 |
| **SamplingUnit** | 137 | 132 | 0 | 0 | 2 |
| **KvCache** | 113 | 549 | 8 | 0 | 0 |
| **Int8MacArray** | 10 | 10 | 0 | 0 | 8 |
| **AxiLiteSlave** | 96 | 173 | 0 | 0 | 0 |
| **ZyboGPTTop** (total) | 15,263 | 12,638 | 25 | 11 | 67 |
| AXI interconnect | 325 | 416 | 0 | 0 | 0 |
| PS7 + reset | 39 | 25 | 0 | 0 | 0 |
| **system_wrapper** (total) | 15,627 | 13,079 | 25 | 11 | 67 |

Notes:
- `Flow_AreaOptimized_high` synthesis maps multiply chains to DSP48E1 aggressively
- Int8MacUnit: 1 DSP each (8 total), fully absorbed (0 LUTs, 0 FFs per unit)
- WeightDecoder: 5 DSPs each (r×171 multiply stages)
- Attention: 32 DSPs (dot product arithmetic)
- Placed LUTs (14,952) < synthesis LUTs (15,263) due to optimization during placement

## Optimization History

| Stage | LUTs | Change | Key Technique |
|-------|------|--------|---------------|
| Initial synthesis | 128,000 | — | Naive Vec(Reg()) everywhere |
| Softmax BRAM conversion | 85,000 | -34% | scoreBuf, expBuf → Mem() |
| All BRAM conversions | 52,000 | -39% | RMSNorm, Attention, FFN, Embedding buffers |
| Phase 1-3 (serialize) | 21,000 | -60% | 1 TDotUnit, compile-time unrolling |
| Phase 4-6 (interfaces) | 17,500 | -17% | Address/data interfaces, KV shifts |
| MAC serialization | 15,000 | -14% | 8 shared MACs, parallel logit |

## Root Causes of Original 128K LUTs

### Vec(Reg()) mux trees

SpinalHDL `Vec(Reg())` with variable-indexed access generates N-to-1 multiplexer
trees. A 128-element Vec indexed by `step` creates 128 comparators + cascade of
2-to-1 muxes. Converting to `Mem()` (BRAM) with sequential access eliminates these.

Buffers converted:
- Softmax scoreBuf (128×16), expBuf (128×16), resultBuf (128×8)
- Attention scoreBuf (128×16)
- FeedForward upBuf (256×8)
- RMSNorm xBuf (64×8)
- Embedding outputBuf (64×8)
- Gamma registers (3×64×16) → single normGammaMem

### Parallel compute units

32 parallel TDotUnits + 32 parallel combinational dot products consumed most of
the remaining LUTs. Replaced with 1 time-multiplexed TDotUnit and 8 shared
INT8 MACs.

### Combinational arithmetic

Wide adder trees, combinational multiply-accumulate chains, and O(N)-depth
carry chains. Resolved through pipelining and DSP inference.

## 6-Phase LUT Optimization Detail

**Phase 1 (FeedForward)**: Compile-time constant indexing in DOWN_PROJ —
unrolled batchIdx eliminates 32×64-to-1 mux trees.

**Phase 2 (WeightBram)**: Single TDotUnit + decodedBankMem replaces 32 parallel
TDotUnits + 103-bank register file.

**Phase 3 (Attention)**: Unrolled headIdx (nHeads=2) in ATTN_SCORE/ATTN_VALUE/
STORE_KV with compile-time constant indices. Serialized dot products via 8 MACs.

**Phase 4 (Softmax)**: resultBuf Vec → probMem BRAM, io.probs Vec → addr/data
interface (probAddr/probData).

**Phase 5 (KvCache)**: Replaced multiply-based address computation with bit
shifts (<<13, <<12, <<5). Eliminated writeBufK/V staging registers.

**Phase 6 (Embedding)**: outputBuf Vec → outputMem BRAM, io.embedding Vec →
addr/data interface. Added EMB_READ state to Sequencer for serial readout.
