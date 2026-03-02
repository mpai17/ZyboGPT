# Create Vivado project for ZyboGPT on Zybo Z7-10
# Usage: vivado -mode batch -source vivado/create_project.tcl

set project_name "zybogpt"
set project_dir "vivado/project"
set part "xc7z010clg400-1"
set board "digilentinc.com:zybo-z7-10:part0:1.2"

# Create project
create_project $project_name $project_dir -part $part -force
set_property board_part $board [current_project]

# Add SpinalHDL-generated Verilog
add_files -norecurse [glob hw/gen/*.v]

# Add constraints
add_files -fileset constrs_1 -norecurse hw/constraints/zybo_z7_10.xdc

# Add weight initialization files (baked into Verilog at elaboration,
# but included here for reference/documentation)
foreach memfile {weights_ternary_32b.mem tok_emb_16b.mem pos_emb_16b.mem} {
    if {[file exists export/$memfile]} {
        add_files -norecurse export/$memfile
    }
}

# Source block design
source vivado/block_design.tcl

# Synthesize BD modules inline (not OOC) to inherit top-level area strategy
set_property SYNTH_CHECKPOINT_MODE None [get_files system.bd]

# Create HDL wrapper
make_wrapper -files [get_files */sources_1/bd/system/system.bd] -top
add_files -norecurse [glob $project_dir/$project_name.gen/sources_1/bd/system/hdl/system_wrapper.v]
set_property top system_wrapper [current_fileset]

# Set synthesis strategy for area optimization (free DSP mapping).
set_property strategy Flow_AreaOptimized_high [get_runs synth_1]

# Set performance-focused implementation strategy
set_property strategy Performance_Explore [get_runs impl_1]

puts "Project created successfully: $project_dir/$project_name.xpr"
puts "Open with: vivado $project_dir/$project_name.xpr"
