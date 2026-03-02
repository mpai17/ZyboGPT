# Run implementation on existing ZyboGPT project
# Usage: vivado -mode batch -source vivado/run_impl.tcl

open_project vivado/project/zybogpt.xpr

# Use aggressive performance strategy with remap optimization for better timing closure.
# (Performance_ExplorePostRoutePhysOpt segfaults in librdi_route.so — avoid it)
set_property strategy Performance_ExploreWithRemap [get_runs impl_1]

set num_jobs [exec nproc]
reset_run impl_1
launch_runs impl_1 -jobs $num_jobs
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "Implementation status: $impl_status"
# Accept any completed status (route_design or phys_opt_design depending on strategy)
if {![string match "*Complete*" $impl_status]} {
    puts "ERROR: Implementation failed! Status: $impl_status"
    exit 1
}
puts "Implementation completed successfully."
