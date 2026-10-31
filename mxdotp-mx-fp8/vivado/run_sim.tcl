# ------------------------------------------------------------------------------
# run_sim.tcl : runs BOTH testbenches in xsim (batch) and prints PASS/FAIL.
#   vivado -mode batch -source vivado/run_sim.tcl             (from the repo root)
# The vector path is passed with +VEC=<abs path>, so no file-copy tricks are needed.
# ------------------------------------------------------------------------------
set root [file normalize [file join [file dirname [info script]] ..]]
if {[llength [get_projects -quiet]] == 0} {
    source $root/vivado/create_project.tcl
}
foreach {top vec} [list tb_mxdotp_unit unit_vectors.mem  tb_mxdotp_blk32 blk_vectors.mem] {
    set_property top $top [get_filesets sim_1]
    set_property -name {xsim.simulate.xsim.more_options} \
                 -value "-testplusarg VEC=$root/sim/vectors/$vec" -objects [get_filesets sim_1]
    set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "=============== $top ==============="
    launch_simulation
    close_sim
}
