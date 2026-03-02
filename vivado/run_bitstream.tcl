# Generate bitstream for ZyboGPT project
# Usage: vivado -mode batch -source vivado/run_bitstream.tcl

open_project vivado/project/zybogpt.xpr
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

puts "Bitstream generated at: vivado/project/zybogpt.runs/impl_1/system_wrapper.bit"
