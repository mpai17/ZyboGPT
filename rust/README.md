# ZyboGPT Rust Firmware

Bare-metal firmware for the Zynq-7010 ARM Cortex-A9 that drives the ZyboGPT FPGA accelerator and communicates with a host over UART.

## Overview

The firmware runs on the Zynq PS (Processing System) without an OS. It initializes the UART and accelerator, accepts text prompts via serial or XSDB mailbox, feeds tokens to the PL (Programmable Logic) accelerator via AXI-Lite MMIO registers, and streams generated text back to the host.

### What It Does

1. Initialize UART1 (115200 baud) and accelerator
2. Print model configuration
3. Enter command loop:
   - Receive prompt text over UART or XSDB mailbox
   - Feed prompt tokens to accelerator (prefill)
   - Autoregressive generation: read output token, feed it back
   - Stream generated characters to host
   - Report performance (tokens/sec, cycle count)

## Prerequisites

- **Rust nightly toolchain** (for Tier 3 `armv7a-none-eabihf` target)
- **rust-src** component (for `-Z build-std`)

These are configured automatically by `rust-toolchain.toml`. On first build, rustup will install the nightly toolchain if needed:

```bash
# If needed manually:
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
```

## Building

```bash
# From project root
make rust

# Or directly:
cd rust && cargo build --release
```

**Output:** `rust/target/armv7a-none-eabihf/release/zybogpt-fw` (ELF binary)

The binary is statically linked, runs from On-Chip Memory (OCM), and requires no runtime or OS.

## Deploying

Load the firmware onto the Zynq ARM core using Vivado's XSDB debugger:

```bash
# From project root (programs bitstream + firmware):
make flash
```

Or manually via XSDB:

```tcl
# In XSDB (after programming the FPGA bitstream):
connect
targets -set -filter {name =~ "APU*"}
dow rust/target/armv7a-none-eabihf/release/zybogpt-fw
con
```

## Interactive Console

```bash
make console
```

This launches `scripts/cmd.py`, which auto-detects whether UART RX works. If not, it falls back to an XSDB mailbox at OCM address `0xFFFFF000`.

## Serial Protocol

Connect to the Zybo's USB-UART at **115200 baud, 8N1** (e.g., `picocom -b 115200 /dev/ttyUSB1`).

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `<text>` | Generate text from prompt | `Hello` |
| `PROMPT:<text>` | Generate text (explicit prefix) | `PROMPT:To be or` |
| `CONFIG` | Print model configuration | `CONFIG` |
| `BENCH` | Run 128-token benchmark | `BENCH` |

### Response Format

```
> Hello                          # User input
GEN:Hello world, I say...       # Generated text (prompt echoed + generation)
STATS:tokens=123,cycles=287820,tok_per_sec=3072
```

## Source Files

| File | Description |
|------|-------------|
| `src/main.rs` | Entry point (`_start` asm stub + `main`), generation loop, benchmark |
| `src/accelerator.rs` | MMIO register interface to PL accelerator (0x43C0_0000) |
| `src/uart.rs` | Zynq PS UART1 driver (0xE000_1000), 115200 8N1 |
| `src/protocol.rs` | Command parsing and response formatting |
| `src/tokenizer.rs` | ASCII char-level tokenizer (0-127) |

### Configuration Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Package config, dependencies (`volatile-register`, `panic-halt`) |
| `rust-toolchain.toml` | Pins nightly toolchain, includes `rust-src` |
| `.cargo/config.toml` | Target (`armv7a-none-eabihf`), linker script, `build-std = ["core"]` |
| `memory.x` | Linker memory regions and sections |
| `linker.x` | Includes `memory.x` |

## Memory Layout

```
0x0000_0000 +---------------------+
            | .text.startup       |  _start (SP init + branch to main)
            | .text               |  Code
            | .rodata             |  Constants, strings
            | .data               |  Initialized globals
            | .bss                |  Zero-initialized globals
            |                     |
            | .stack (16 KB)      |  <- SP initialized to __stack_top
0x0004_0000 +---------------------+  End of OCM (256 KB)

0xFFFF_F000 +---------------------+
            | XSDB mailbox        |  cmd_ready, cmd_len, cmd_data[256]
0xFFFF_FFFF +---------------------+  High OCM (non-cacheable)
```

The firmware runs entirely from the 256 KB On-Chip Memory.

## Accelerator Register Interface

Base address: `0x43C0_0000` (Zynq AXI GP0)

```rust
#[repr(C)]
struct AccelRegs {
    control:   RW<u32>,  // 0x00 - [0]=start, [1]=reset, [3:2]=mode
    status:    RO<u32>,  // 0x04 - [0]=busy, [1]=done
    token_in:  WO<u32>,  // 0x08 - [6:0]=input token
    token_out: RO<u32>,  // 0x0C - [6:0]=output token
    position:  WO<u32>,  // 0x10 - [6:0]=sequence position
    cycle_lo:  RO<u32>,  // 0x14 - cycle counter [31:0]
    cycle_hi:  RO<u32>,  // 0x18 - cycle counter [63:32]
    config:    RO<u32>,  // 0x1C - model config packed
    sampling:  RW<u32>,  // 0x20 - [15:0]=inv_temp
    seed:      RW<u32>,  // 0x24 - [31:0]=LFSR seed
}
```

### Inference Sequence

```rust
// 1. Write input token and position
regs.token_in.write(token & 0x7F);
regs.position.write(position & 0x7F);

// 2. Pulse start bit (rising edge)
regs.control.write(0x00);
regs.control.write(0x01);
regs.control.write(0x00);

// 3. Poll for completion
loop {
    if regs.status.read() & 0x02 != 0 { break; }  // Done bit
}

// 4. Read result
let output_token = regs.token_out.read() & 0x7F;
let cycles = regs.cycle_lo.read();
```

## Bare-Metal Startup

The firmware uses `#![no_std]` and `#![no_main]`. A minimal assembly stub in `main.rs` provides the entry point:

```asm
.section .text.startup
.global _start
_start:
    ldr sp, =__stack_top    @ Set stack pointer from linker script
    b main                  @ Branch to Rust main
```

The `panic_halt` crate provides the panic handler (infinite loop on panic).

## UART Configuration

UART1 is configured for 115200 baud using the Zynq PS UART controller:

| Parameter | Value |
|-----------|-------|
| Base address | 0xE000_1000 |
| Baud rate | 115200 |
| Ref clock | 100 MHz (Zynq PS IO PLL) |
| Clock divisor (CD) | 124 |
| Baud divider (BDIV) | 6 |
| Data bits | 8 |
| Parity | None |
| Stop bits | 1 |

Formula: `baud = 100_000_000 / (124 * (6+1)) = 115,207`

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `volatile-register` | 0.2 | Safe MMIO register access (`RO`, `RW`, `WO` types) |
| `panic-halt` | 1.0 | Panic handler (halts on panic) |

Release profile: size-optimized (`opt-level = "s"`), LTO enabled, single codegen unit.
