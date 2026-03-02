/* Zynq-7010 memory map for bare-metal Rust firmware */

MEMORY
{
    /* On-chip memory (OCM): 256 KB */
    OCM (rwx)   : ORIGIN = 0x00000000, LENGTH = 256K

    /* DDR3L: 1 GB (shared with Linux if running, exclusive in bare-metal) */
    DDR (rwx)   : ORIGIN = 0x00100000, LENGTH = 512M

    /* PL AXI GP0 address space */
    /* ACCEL (rw) : ORIGIN = 0x43C00000, LENGTH = 4K */
}

SECTIONS
{
    .text : {
        *(.text.startup)
        *(.text*)
    } > OCM

    .rodata : {
        *(.rodata*)
    } > OCM

    .data : {
        *(.data*)
    } > OCM

    .bss (NOLOAD) : {
        *(.bss*)
    } > OCM

    .heap (NOLOAD) : {
        . = ALIGN(8);
        __heap_start = .;
        . += 64K;
        __heap_end = .;
    } > DDR

    .stack (NOLOAD) : {
        . = ALIGN(8);
        . += 16K;
        __stack_top = .;
    } > OCM
}

ENTRY(_start)
