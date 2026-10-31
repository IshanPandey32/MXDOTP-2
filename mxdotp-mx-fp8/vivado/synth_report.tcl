# ------------------------------------------------------------------------------
# synth_report.tcl : out-of-context synthesis of BASELINE vs IMPROVED (area / timing)
#   vivado -mode batch -source vivado/synth_report.tcl        (from the repo root)
# Reports land in vivado/reports/.  Use them to fill the results table in the README.
# ------------------------------------------------------------------------------
set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir $root/vivado/reports
set part xc7a100tcsg324-1
set files [glob $root/rtl/common/*.v $root/rtl/baseline/*.v $root/rtl/improved/*.v]

foreach top {mxdotp_unit mxdotp_blk32} {
    read_verilog $files
    synth_design -top $top -part $part -mode out_of_context
    create_clock -period 10.0 -name clk [get_ports clk]
    report_utilization    -file $root/vivado/reports/${top}_utilization.rpt
    report_timing_summary -file $root/vivado/reports/${top}_timing.rpt
    report_power          -file $root/vivado/reports/${top}_power.rpt
    close_design
}
