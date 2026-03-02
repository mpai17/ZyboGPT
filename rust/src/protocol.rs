//! Simple text protocol for host communication over UART.
//!
//! Protocol:
//!   Host sends: "PROMPT:<text>\n"
//!   Device responds: "GEN:<generated_text>\n"
//!   Device sends: "STATS:tokens=N,cycles=M,tok_per_sec=K\n"
//!
//! Special commands:
//!   "CONFIG\n" - Print model configuration
//!   "BENCH\n"  - Run 128-token benchmark

use crate::uart::Uart;

const MAX_PROMPT_LEN: usize = 128;

pub enum Command {
    Prompt { text: [u8; MAX_PROMPT_LEN], len: usize },
    Config,
    Bench,
    Unknown,
}

/// Read a line from UART into buffer. Returns length.
pub fn read_line(uart: &Uart, buf: &mut [u8]) -> usize {
    let mut i = 0;
    loop {
        let b = uart.read_byte_blocking();
        if b == b'\n' || b == b'\r' {
            break;
        }
        if i < buf.len() {
            buf[i] = b;
            i += 1;
        }
    }
    i
}

/// Parse a command from input line.
pub fn parse_command(buf: &[u8], len: usize) -> Command {
    if len == 0 {
        return Command::Unknown;
    }

    // Check for special commands
    if len == 6 && &buf[..6] == b"CONFIG" {
        return Command::Config;
    }
    if len == 5 && &buf[..5] == b"BENCH" {
        return Command::Bench;
    }

    // Check for PROMPT: prefix
    if len > 7 && &buf[..7] == b"PROMPT:" {
        let mut text = [0u8; MAX_PROMPT_LEN];
        let text_len = (len - 7).min(MAX_PROMPT_LEN);
        text[..text_len].copy_from_slice(&buf[7..7 + text_len]);
        return Command::Prompt {
            text,
            len: text_len,
        };
    }

    // Treat bare text as prompt
    let mut text = [0u8; MAX_PROMPT_LEN];
    let text_len = len.min(MAX_PROMPT_LEN);
    text[..text_len].copy_from_slice(&buf[..text_len]);
    Command::Prompt {
        text,
        len: text_len,
    }
}

/// Send generation stats over UART.
pub fn send_stats(uart: &mut Uart, num_tokens: u32, total_cycles: u32, clock_mhz: u32) {
    let tok_per_sec = if total_cycles > 0 {
        (num_tokens as u64 * clock_mhz as u64 * 1_000_000) / total_cycles as u64
    } else {
        0
    };

    uart.write_str("STATS:tokens=");
    uart.write_u32(num_tokens);
    uart.write_str(",cycles=");
    uart.write_u32(total_cycles);
    uart.write_str(",tok_per_sec=");
    uart.write_u32(tok_per_sec as u32);
    uart.write_str("\n");
}
