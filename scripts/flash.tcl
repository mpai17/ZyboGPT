# flash.tcl - Program Zybo Z7-10 with bitstream + bare-metal firmware
# Usage: xsdb flash.tcl <bitstream> <firmware_elf> <ps7_init_tcl>
#
# Correct sequence for Zynq bare-metal:
#   1. Connect and system reset (recovers DAP if in error state)
#   2. PS7 init (clocks, MIO, UART) while CPU is stopped
#   3. Program FPGA bitstream (MUST be after rst -system, which clears PL)
#   4. Download firmware ELF to OCM
#   5. Start execution

set bitstream [lindex $argv 0]
set firmware  [lindex $argv 1]
set ps7_init  [lindex $argv 2]

puts "Bitstream: $bitstream"
puts "Firmware:  $firmware"
puts "PS7 init:  $ps7_init"

# Step 1: Connect and system reset
connect
after 3000
puts "=== Targets after connect ==="
puts [targets]

# System reset to recover DAP and get clean state
puts "=== System reset ==="
catch {
    targets -set -nocase -filter {name =~ "APU*" || name =~ "DAP*"}
    rst -system
} rst_result
puts "rst result: $rst_result"
after 5000

puts "=== Targets after reset ==="
puts [targets]

# Step 2: Select CPU #0 and stop it
puts "=== Selecting ARM Cortex-A9 #0 ==="
targets -set -nocase -filter {name =~ "*Cortex*#0" || name =~ "*ARM*#0"}
catch {stop}
after 500

# Step 3: Initialize PS7 (clocks, DDR, MIO, UART)
puts "=== Initializing PS7 ==="
source $ps7_init
ps7_init
ps7_post_config
after 1000

# Step 4: Program FPGA bitstream (AFTER ps7_init, so PL clocks are running)
puts "=== Programming FPGA ==="
targets -set -filter {name =~ "xc7z*"}
fpga $bitstream
after 3000

# Step 5: Download firmware to OCM
puts "=== Downloading firmware ==="
targets -set -nocase -filter {name =~ "*Cortex*#0" || name =~ "*ARM*#0"}
dow $firmware
after 1000

# Step 6: Start execution
puts "=== Starting execution ==="
con
after 2000

puts "=== Done! Connect to the Zybo serial port at 115200 baud ==="
disconnect
