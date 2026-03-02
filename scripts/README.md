# ZyboGPT Scripts

Helper scripts for the ZyboGPT build pipeline, board interaction, and development tools.

## Pipeline Scripts

These scripts are called by `make` targets and form the core build pipeline.

| Script | Makefile Target | Description |
|--------|----------------|-------------|
| `generate_weights.py` | `make export` | Converts PyTorch checkpoint to BRAM init files (.mem, .coe), generates `WeightInit.scala`, copies it to `hw/src/main/scala/zybogpt/` |
| `validate_model.py` | `make validate` | Compares FP32 vs INT8 inference, generates RTL test vectors to `export/test_vectors/` |
| `update_test_refs.py` | (manual) | Updates Scala test reference arrays in `ZyboGPTRomeoSim.scala` and `ZyboGPTPipelineDebugSim.scala` from `test_vectors.json` |
| `flash.tcl` | `make flash` | XSDB script to program bitstream + firmware onto Zybo Z7-10 via JTAG |

## Interactive Tools

| Script | Makefile Target | Description |
|--------|----------------|-------------|
| `board.py` | (library) | Shared board utilities: cross-platform serial port detection using pyserial `list_ports` |
| `cmd.py` | `make console` | Interactive ollama-style console for ZyboGPT. Auto-detects UART RX; falls back to XSDB mailbox if UART RX is broken. Supports single-command mode: `python scripts/cmd.py BENCH` |
| `benchmark.py` | `make benchmark` | Cross-platform throughput comparison (CPU vs GPU vs FPGA). Connects to board via serial for FPGA results. |
| `generate.py` | (manual) | Generate text from a trained model using float PyTorch inference. Usage: `python scripts/generate.py --prompt "ROMEO:" --temperature 0.5` |

## Usage

### Export weights after training

```bash
make export
# Equivalent to:
venv/bin/python scripts/generate_weights.py checkpoints/phase2/best.pt --output export
```

### Validate quantized model

```bash
make validate
# Equivalent to:
venv/bin/python scripts/validate_model.py checkpoints/phase2/best.pt --export export
```

### Update test references after retraining

After retraining the model and re-exporting, update the hardcoded reference arrays in the SpinalHDL test files:

```bash
venv/bin/python scripts/update_test_refs.py
# Then verify:
make spinal-test
```

### Interactive console

```bash
make console
# Commands: CONFIG, BENCH, or any text prompt
```

`cmd.py` locates XSDB automatically (via `PATH` or common install directories). To use a
custom install path, set the `XSDB` environment variable:

```bash
export XSDB=/path/to/Vivado/2024.1/bin/xsdb
```

### Flash board

```bash
make flash
# Programs bitstream + firmware via XSDB
```

### Benchmark (CPU vs GPU vs FPGA)

```bash
make benchmark
# Or with options:
venv/bin/python scripts/benchmark.py --no-fpga          # skip FPGA (no board needed)
venv/bin/python scripts/benchmark.py --fpga-only         # only FPGA
venv/bin/python scripts/benchmark.py --gen-tokens 128    # more tokens per run
```

### Generate text (float model)

```bash
venv/bin/python scripts/generate.py --prompt "To be" --tokens 100 --temperature 0.5
venv/bin/python scripts/generate.py --checkpoint checkpoints/phase1/best.pt
```
