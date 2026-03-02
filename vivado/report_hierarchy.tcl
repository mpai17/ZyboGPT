# Generate hierarchical utilization report from synthesized design
# Usage: vivado -mode batch -source vivado/report_hierarchy.tcl

open_project vivado/project/zybogpt.xpr
open_run synth_1

# Deep hierarchical utilization - depth 8 to reach SpinalHDL sub-modules
report_utilization -hierarchical -hierarchical_depth 8 -file vivado/reports/hier_util.rpt

# Try per-cell reports for the accelerator
set accel_cell [get_cells -hierarchical -filter {ORIG_REF_NAME == ZyboGPTTop || REF_NAME == ZyboGPTTop || REF_NAME =~ *ZyboGPTTop}]
if {$accel_cell ne ""} {
    puts "Found accelerator cell: $accel_cell"
    report_utilization -hierarchical -hierarchical_depth 6 -cells $accel_cell -file vivado/reports/accel_hier_util.rpt
} else {
    puts "ZyboGPTTop cell not found, listing all cells..."
}

# List all hierarchical cells at depth to see what's available
foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == false}] {
    puts "CELL: $cell  REF: [get_property REF_NAME $cell]"
}

puts "Reports written to vivado/reports/"
