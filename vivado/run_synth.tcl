# Run synthesis on existing ZyboGPT project
# Usage: vivado -mode batch -source vivado/run_synth.tcl

open_project vivado/project/zybogpt.xpr

set num_jobs [exec nproc]
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1

if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}
puts "Synthesis completed successfully."

# Post-synthesis static timing analysis and utilization
file mkdir vivado/reports
open_run synth_1

puts "\n===== POST-SYNTHESIS UTILIZATION ====="
report_utilization -hierarchical -hierarchical_depth 2 -file vivado/reports/synth_utilization.rpt
report_utilization

puts "\n===== POST-SYNTHESIS TIMING (top 20 worst paths) ====="
report_timing_summary -file vivado/reports/synth_timing_summary.rpt
report_timing -max_paths 20 -sort_by slack -file vivado/reports/synth_timing_paths.rpt
report_timing_summary -no_header -no_detailed_paths

puts "\n===== DSP USAGE ====="
set dsp_count [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP.*}]]
puts "Total DSP48E1 cells: $dsp_count"
foreach dsp [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP.*}] {
    puts "  DSP: $dsp"
}
