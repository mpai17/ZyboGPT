//! MMIO register interface to ZyboGPT PL accelerator.
//!
//! Base address: 0x43C0_0000 (AXI GP0)
//!
//! Register map:
//!   0x00 CONTROL   [W]  bit0: start, bit[3:2]: mode
//!   0x04 STATUS    [R]  bit0: busy, bit1: done, [31:16]: cycle_count
//!   0x08 TOKEN_IN  [W]  [6:0]: input token ID
//!   0x0C TOKEN_OUT [R]  [6:0]: output token ID
//!   0x10 POSITION  [W]  [6:0]: current sequence position
//!   0x14 CYCLE_LO  [R]  cycle counter [31:0]
//!   0x18 CYCLE_HI  [R]  cycle counter [63:32]
//!   0x1C CONFIG    [R]  model config

use volatile_register::{RO, RW, WO};

const ACCEL_BASE: usize = 0x43C0_0000;

#[repr(C)]
struct AccelRegs {
    control: RW<u32>,   // 0x00
    status: RO<u32>,    // 0x04
    token_in: WO<u32>,  // 0x08
    token_out: RO<u32>, // 0x0C
    position: WO<u32>,  // 0x10
    cycle_lo: RO<u32>,  // 0x14
    cycle_hi: RO<u32>,  // 0x18
    config: RO<u32>,    // 0x1C
    sampling: RW<u32>,  // 0x20 [15:0]: inv_temp (0 = greedy argmax)
    seed: RW<u32>,      // 0x24 [31:0]: LFSR seed for temperature sampling
}

pub struct Accelerator {
    regs: &'static mut AccelRegs,
}

impl Accelerator {
    /// Create accelerator interface. Unsafe: caller must ensure single instance.
    pub unsafe fn new() -> Self {
        Self {
            regs: &mut *(ACCEL_BASE as *mut AccelRegs),
        }
    }

    /// Run single-token inference.
    /// Returns (output_token, cycle_count).
    pub fn infer_token(&mut self, token: u8, position: u8) -> (u8, u32) {
        unsafe {
            // Set input token and position
            self.regs.token_in.write(token as u32 & 0x7F);
            self.regs.position.write(position as u32 & 0x7F);

            // Trigger start (rising edge)
            self.regs.control.write(0x00);
            self.regs.control.write(0x01);
            self.regs.control.write(0x00);

            // Wait for completion
            loop {
                let status = self.regs.status.read();
                if status & 0x02 != 0 {
                    // Done bit set
                    break;
                }
                core::hint::spin_loop();
            }

            let token_out = (self.regs.token_out.read() & 0x7F) as u8;
            let cycles = self.regs.cycle_lo.read();

            (token_out, cycles)
        }
    }

    /// Check if accelerator is busy.
    pub fn is_busy(&self) -> bool {
        self.regs.status.read() & 0x01 != 0
    }

    /// Read accelerator config register.
    pub fn read_config(&self) -> AccelConfig {
        let val = self.regs.config.read();
        AccelConfig {
            d_model: (val & 0xFF) as u8,
            n_layers: ((val >> 8) & 0xFF) as u8,
            ctx_len: ((val >> 16) & 0xFF) as u8,
            vocab_size: ((val >> 24) & 0xFF) as u8,
        }
    }

    /// Read cycle counter.
    pub fn read_cycles(&self) -> u32 {
        self.regs.cycle_lo.read()
    }

    /// Configure temperature sampling.
    /// inv_temp: inverse temperature (0 = greedy argmax, 512 = T=0.5)
    /// seed: LFSR seed for deterministic randomness
    pub fn set_sampling(&mut self, inv_temp: u16, seed: u32) {
        unsafe {
            self.regs.seed.write(seed);      // Must be set before inv_temp
            self.regs.sampling.write(inv_temp as u32);
        }
    }
}

#[derive(Debug)]
pub struct AccelConfig {
    pub d_model: u8,
    pub n_layers: u8,
    pub ctx_len: u8,
    pub vocab_size: u8,
}
