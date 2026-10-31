# ------------------------------------------------------------------------------
# create_project.tcl : creates a Vivado project with all RTL + testbenches.
#   vivado -mode batch -source vivado/create_project.tcl      (from the repo root)
#   or GUI: Tools > Run Tcl Script...
# Generate the vectors first:  python3 scripts/gen_vectors.py
# ------------------------------------------------------------------------------
set root [file normalize [file join [file dirname [info script]] ..]]
set proj mxdotp_proj

create_project $proj $root/vivado/$proj -part xc7a100tcsg324-1 -force

add_files -norecurse [glob $root/rtl/common/*.v $root/rtl/baseline/*.v $root/rtl/improved/*.v]
add_files -fileset sim_1 -norecurse [glob $root/tb/*.v]
add_files -fileset sim_1 -norecurse [glob -nocomplain $root/sim/vectors/*.mem]

set_property top mxdotp_unit [current_fileset]
set_property top tb_mxdotp_unit [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
puts "Project created. Run simulation with: vivado -mode batch -source vivado/run_sim.tcl"
