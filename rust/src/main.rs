//! ZyboGPT bare-metal firmware for Zynq-7010 ARM Cortex-A9.
//!
//! Entry point for the generation loop:
//! 1. Initialize UART and accelerator
//! 2. Wait for prompt from host
//! 3. Feed prompt tokens to accelerator
//! 4. Stream generated tokens back over UART
//! 5. Report performance statistics

#![no_std]
#![no_main]

mod accelerator;
mod protocol;
mod tokenizer;
mod uart;

use accelerator::Accelerator;
use protocol::{parse_command, send_stats, Command};
use tokenizer::Tokenizer;
use uart::Uart;

extern crate panic_halt;

const CLOCK_MHZ: u32 = 150;
const MAX_GEN_TOKENS: usize = 128;

// XSDB mailbox in high OCM for host→device commands (backup for broken UART RX).
// Host writes here via XSDB `mwr` (scripts/cmd.py), firmware polls it.
// OCM section 3 is at high addresses (OCM_CFG=0x18): 0xFFFF0000-0xFFFFFFFF.
// OCM is non-cacheable, so JTAG DAP writes are immediately visible to the CPU.
const MAILBOX_BASE: usize = 0xFFFF_F000;
// Mailbox layout: +0x00 cmd_ready(u32), +0x04 cmd_len(u32), +0x08 cmd_data(256 bytes)

unsafe fn mailbox_check() -> Option<usize> {
    let ready = core::ptr::read_volatile(MAILBOX_BASE as *const u32);
    if ready == 1 {
        let len = core::ptr::read_volatile((MAILBOX_BASE + 4) as *const u32) as usize;
        Some(len.min(256))
    } else {
        None
    }
}

unsafe fn mailbox_read(buf: &mut [u8], len: usize) {
    let src = (MAILBOX_BASE + 8) as *const u8;
    let n = len.min(buf.len());
    for i in 0..n {
        buf[i] = core::ptr::read_volatile(src.add(i));
    }
    core::ptr::write_volatile(MAILBOX_BASE as *mut u32, 0);
}

unsafe fn mailbox_init() {
    core::ptr::write_volatile(MAILBOX_BASE as *mut u32, 0);
}

// Bare-metal startup: set stack pointer and branch to main.
// Placed in .text.startup so the linker puts it first (see memory.x).
core::arch::global_asm!(
    ".section .text.startup, \"ax\"",
    ".global _start",
    "_start:",
    "    ldr sp, =__stack_top",
    "    b main",
);

#[no_mangle]
pub unsafe extern "C" fn main() -> ! {
    let mut uart = Uart::new();
    let mut accel = Accelerator::new();

    uart.write_str("\n=== ZyboGPT v0.1 ===\n");
    uart.write_str("Ternary LLM on Zybo Z7-10 FPGA\n");

    // Print config
    let config = accel.read_config();
    uart.write_str("Model: d_model=");
    uart.write_u32(config.d_model as u32);
    uart.write_str(", layers=");
    uart.write_u32(config.n_layers as u32);
    uart.write_str(", ctx=");
    uart.write_u32(config.ctx_len as u32);
    uart.write_str(", vocab=");
    uart.write_u32(config.vocab_size as u32);
    uart.write_str("\n");

    // Configure T=0.5 sampling
    accel.set_sampling(512, 0xDEADBEEF); // inv_temp=512 (T=0.5), deterministic seed
    uart.write_str("Accelerator ready (T=0.5 sampling).\n\n");

    // Initialize XSDB mailbox (backup for broken UART RX)
    mailbox_init();

    // Auto-run: generate from "ROMEO:" prompt immediately
    uart.write_str("Auto-generating from ROMEO:...\n");
    generate(&mut uart, &mut accel, b"ROMEO:");
    uart.write_str("\nGeneration complete.\n");

    let mut line_buf = [0u8; 256];

    uart.write_str("> ");
    uart.flush();

    loop {
        // Poll UART RX and XSDB mailbox (whichever has data first)
        let cmd = if let Some(len) = mailbox_check() {
            mailbox_read(&mut line_buf, len);
            uart.write_str("[XSDB] ");
            for i in 0..len { uart.write_byte(line_buf[i]); }
            uart.write_str("\n");
            parse_command(&line_buf, len)
        } else if let Some(b) = uart.read_byte() {
            line_buf[0] = b;
            let mut i = 1usize;
            if b != b'\n' && b != b'\r' {
                loop {
                    let nb = uart.read_byte_blocking();
                    if nb == b'\n' || nb == b'\r' { break; }
                    if i < line_buf.len() { line_buf[i] = nb; i += 1; }
                }
            } else {
                i = 0;
            }
            parse_command(&line_buf, i)
        } else {
            for _ in 0..1000 { core::hint::spin_loop(); }
            continue;
        };

        match cmd {
            Command::Prompt { text, len: plen } => {
                generate(&mut uart, &mut accel, &text[..plen]);
            }
            Command::Config => {
                let c = accel.read_config();
                uart.write_str("d_model=");
                uart.write_u32(c.d_model as u32);
                uart.write_str(" n_layers=");
                uart.write_u32(c.n_layers as u32);
                uart.write_str(" ctx_len=");
                uart.write_u32(c.ctx_len as u32);
                uart.write_str(" vocab=");
                uart.write_u32(c.vocab_size as u32);
                uart.write_str("\n");
            }
            Command::Bench => {
                benchmark(&mut uart, &mut accel);
            }
            Command::Unknown => {
                uart.write_str("Unknown command. Send text or: CONFIG, BENCH\n");
            }
        }
        uart.write_str("> ");
        uart.flush();
    }
}

/// Run text generation from a prompt (with stats output).
fn generate(uart: &mut Uart, accel: &mut Accelerator, prompt: &[u8]) {
    uart.write_str("GEN:");
    let (gen_count, total_cycles) = generate_counted(uart, accel, prompt);
    send_stats(uart, gen_count, total_cycles, CLOCK_MHZ);
}

/// Generate from a prompt, returning (generated_token_count, total_cycles).
fn generate_counted(uart: &mut Uart, accel: &mut Accelerator, prompt: &[u8]) -> (u32, u32) {
    for &b in prompt {
        uart.write_byte(b);
    }

    let mut total_cycles: u32 = 0;
    let mut position: u8 = 0;
    let mut last_out: u8 = 0;

    for &b in prompt {
        let token = Tokenizer::encode_char(b);
        let (out, cycles) = accel.infer_token(token, position);
        last_out = out;
        total_cycles += cycles;
        position += 1;
        if position >= 127 { break; }
    }

    let mut last_token = last_out;
    let gen_tokens = MAX_GEN_TOKENS.min(128 - prompt.len());
    let mut gen_count = 0u32;

    for _ in 0..gen_tokens {
        let (next_token, cycles) = accel.infer_token(last_token, position);
        total_cycles += cycles;
        uart.write_byte(Tokenizer::decode_token(next_token) as u8);
        last_token = next_token;
        position += 1;
        gen_count += 1;
        if position >= 127 { break; }
    }

    uart.write_str("\n");
    (gen_count, total_cycles)
}

/// Continuous benchmark: alternate ROMEO:/JULIET: until stopped or 5 min.
/// Send any mailbox command (or Ctrl+C on host) to stop early.
fn benchmark(uart: &mut Uart, accel: &mut Accelerator) {
    uart.write_str("Continuous benchmark (ROMEO:/JULIET:, 5 min max)\n");
    uart.write_str("Send any command to stop.\n\n");

    let prompts: [&[u8]; 2] = [b"ROMEO:", b"JULIET:"];
    let mut prompt_idx: usize = 0;
    let mut total_tokens: u32 = 0;
    let mut total_cycles_hi: u32 = 0; // upper 32 bits
    let mut total_cycles_lo: u32 = 0; // lower 32 bits
    let mut round: u32 = 0;

    // 5 min at 150 MHz = 45,000,000,000 cycles
    // = 10 * 0xFFFF_FFFF + 2,705,032,708 (roughly)
    // Track as split hi:lo to avoid needing u64 in no_std
    let limit_hi: u32 = 10;
    let limit_lo: u32 = 2_705_032_704;

    loop {
        round += 1;
        let prompt = prompts[prompt_idx];
        prompt_idx ^= 1;

        uart.write_str("[");
        uart.write_u32(round);
        uart.write_str("] ");

        let (gen_tokens, gen_cycles) = generate_counted(uart, accel, prompt);
        total_tokens += gen_tokens;

        // Add gen_cycles to 64-bit counter (hi:lo)
        let new_lo = total_cycles_lo.wrapping_add(gen_cycles);
        if new_lo < total_cycles_lo {
            total_cycles_hi += 1; // carry
        }
        total_cycles_lo = new_lo;

        // Check 5-minute limit
        if total_cycles_hi > limit_hi
            || (total_cycles_hi == limit_hi && total_cycles_lo >= limit_lo)
        {
            uart.write_str("\nTime limit reached (5 min).\n");
            break;
        }

        // Check mailbox for stop signal
        unsafe {
            if let Some(len) = mailbox_check() {
                let mut buf = [0u8; 256];
                mailbox_read(&mut buf, len);
                uart.write_str("\nStopped.\n");
                break;
            }
        }
    }

    // Aggregate stats
    uart.write_str("\n--- Benchmark Results ---\n");
    uart.write_str("Rounds: ");
    uart.write_u32(round);
    uart.write_str("\nTokens: ");
    uart.write_u32(total_tokens);
    uart.write_str("\n");

    // tok/s = total_tokens * 150_000_000 / total_cycles
    // Use 64-bit math: total_cycles = hi * 2^32 + lo
    let clk = CLOCK_MHZ as u64 * 1_000_000;
    let total_cycles_64 = (total_cycles_hi as u64) << 32 | total_cycles_lo as u64;
    let tok_per_sec = if total_cycles_64 > 0 {
        (total_tokens as u64 * clk) / total_cycles_64
    } else {
        0
    };
    let elapsed_s = total_cycles_64 / clk;

    uart.write_str("Time: ");
    uart.write_u32(elapsed_s as u32);
    uart.write_str("s\n");
    uart.write_str("Tokens/sec: ");
    uart.write_u32(tok_per_sec as u32);
    uart.write_str("\n");
}
