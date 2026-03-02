## Zybo Z7-10 constraints for ZyboGPT
## Target: xc7z010clg400-1
##
## ZyboGPTTop is a PS-only design (AXI slave, no PL I/O pins).
## Clock comes from PS FCLK_CLK0, not an external PL pin.

## PL clock constraint: 150 MHz from Zynq PS FCLK_CLK0
## (Configured in block design via PCW_FPGA0_PERIPHERAL_FREQMHZ)
create_generated_clock -name pl_clk -source [get_pins */PS7_i/FCLKCLK[0]] -divide_by 1 [get_pins */PS7_i/FCLKCLK[0]]

## Configuration voltage
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## Pblock for ZyboGPT accelerator (optional, for placement guidance)
# create_pblock pblock_accel
# resize_pblock pblock_accel -add {SLICE_X0Y0:SLICE_X51Y49}
# add_cells_to_pblock pblock_accel [get_cells -hierarchical -filter {NAME =~ */ZyboGPTTop*}]
