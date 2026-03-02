# Block design for ZyboGPT: Zynq PS + ZyboGPT PL accelerator
# Zynq PS M_AXI_GP0 (AXI3) -> AXI interconnect -> ZyboGPT (AXI4-Lite)

# Create block design
create_bd_design "system"

# Add Zynq PS — manually export DDR and FIXED_IO
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
make_bd_intf_pins_external [get_bd_intf_pins ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7/FIXED_IO]

# Configure PS
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {150} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {0} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_USE_S_AXI_HP0 {0} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
] [get_bd_cells ps7]

# Add AXI interconnect for AXI3 -> AXI4-Lite protocol conversion
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_ic]

# Add proc_sys_reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst

# Add ZyboGPT accelerator (wrapper with standard AXI-Lite naming)
create_bd_cell -type module -reference ZyboGPTWrapper accel

# Clock connections
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_ic/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_ic/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_ic/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins accel/aclk]

# Reset connections
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins rst/interconnect_aresetn] [get_bd_pins axi_ic/ARESETN]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins axi_ic/S00_ARESETN]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins axi_ic/M00_ARESETN]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins accel/aresetn]

# AXI connections: PS (AXI3) -> interconnect -> accel (AXI4-Lite)
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic/M00_AXI] [get_bd_intf_pins accel/s_axi]

# Assign address: 0x43C0_0000 base, 4K range
assign_bd_address -target_address_space /ps7/Data [get_bd_addr_segs accel/s_axi/reg0] -range 4K -offset 0x43C00000

# Validate and save
validate_bd_design
save_bd_design

puts "Block design created: Zynq PS (150 MHz FCLK) + AXI interconnect + ZyboGPT @ 0x43C0_0000"
