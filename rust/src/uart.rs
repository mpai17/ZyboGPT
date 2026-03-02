//! UART driver for Zynq PS UART1.
//!
//! UART1 is connected to the USB-UART bridge on Zybo Z7.
//! Base address: 0xE0001000
//! Default baud: 115200

use volatile_register::{RO, RW};

const UART1_BASE: usize = 0xE000_1000;

#[repr(C)]
struct UartRegs {
    control: RW<u32>,     // 0x00
    mode: RW<u32>,        // 0x04
    ier: RW<u32>,         // 0x08 Interrupt Enable
    idr: RW<u32>,         // 0x0C Interrupt Disable
    imr: RO<u32>,         // 0x10 Interrupt Mask
    isr: RW<u32>,         // 0x14 Interrupt Status
    baudgen: RW<u32>,     // 0x18 Baud Rate Generator
    rxtout: RW<u32>,      // 0x1C RX Timeout
    rxwm: RW<u32>,        // 0x20 RX FIFO Watermark
    modemcr: RW<u32>,     // 0x24 Modem Control
    modemsr: RO<u32>,     // 0x28 Modem Status
    sr: RO<u32>,          // 0x2C Channel Status
    fifo: RW<u32>,        // 0x30 TX/RX FIFO
    bauddiv: RW<u32>,     // 0x34 Baud Rate Divider
    flowdelay: RW<u32>,   // 0x38 Flow Control Delay
    _reserved: [u32; 2],
    txwm: RW<u32>,        // 0x44 TX FIFO Watermark
}

// Channel status register bits
const SR_TXFULL: u32 = 1 << 4;
const SR_TXEMPTY: u32 = 1 << 3;
const SR_RXEMPTY: u32 = 1 << 1;

pub struct Uart {
    regs: &'static mut UartRegs,
}

impl Uart {
    /// Initialize UART1. Unsafe: caller ensures single instance.
    pub unsafe fn new() -> Self {
        let regs = &mut *(UART1_BASE as *mut UartRegs);

        // Assume UART is already configured by FSBL/bootloader
        // If bare-metal from scratch, configure here:
        // - Reset TX/RX
        // - Set baud rate (115200 with 100 MHz ref clock)
        // - Enable TX/RX

        // Baud rate: 115200 with 100 MHz UART ref clock
        // (ps7_init: IO PLL = 1200 MHz, UART divisor = 12 → 100 MHz)
        // CD = 124, BDIV = 6 -> 100M / (124 * (6+1)) = 115,207 ≈ 115200
        regs.baudgen.write(124);
        regs.bauddiv.write(6);

        // 8N1 mode, normal channel mode
        regs.mode.write(0x20);

        // Enable TX + RX via Xilinx SDK pattern (read-modify-write)
        // EN_DIS_MASK = 0x3C covers bits [5:2]: TX_DIS, TX_EN, RX_DIS, RX_EN
        // Clear all enable/disable bits, then set only enables
        let cr = regs.control.read();
        regs.control.write((cr & !0x3Cu32) | 0x14); // RX_EN(0x04) | TX_EN(0x10)

        Self { regs }
    }

    /// Write a single byte (blocking).
    pub fn write_byte(&mut self, b: u8) {
        // Wait until TX FIFO not full
        while self.regs.sr.read() & SR_TXFULL != 0 {
            core::hint::spin_loop();
        }
        unsafe {
            self.regs.fifo.write(b as u32);
        }
    }

    /// Write a string.
    pub fn write_str(&mut self, s: &str) {
        for b in s.bytes() {
            if b == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(b);
        }
    }

    /// Write an unsigned integer as decimal.
    pub fn write_u32(&mut self, mut val: u32) {
        if val == 0 {
            self.write_byte(b'0');
            return;
        }
        let mut buf = [0u8; 10];
        let mut i = 0;
        while val > 0 {
            buf[i] = b'0' + (val % 10) as u8;
            val /= 10;
            i += 1;
        }
        while i > 0 {
            i -= 1;
            self.write_byte(buf[i]);
        }
    }

    /// Read a byte if available.
    pub fn read_byte(&self) -> Option<u8> {
        if self.regs.sr.read() & SR_RXEMPTY == 0 {
            Some((self.regs.fifo.read() & 0xFF) as u8)
        } else {
            None
        }
    }

    /// Read a byte (blocking).
    pub fn read_byte_blocking(&self) -> u8 {
        loop {
            if let Some(b) = self.read_byte() {
                return b;
            }
            core::hint::spin_loop();
        }
    }

    /// Flush TX FIFO.
    pub fn flush(&self) {
        while self.regs.sr.read() & SR_TXEMPTY == 0 {
            core::hint::spin_loop();
        }
    }
}
