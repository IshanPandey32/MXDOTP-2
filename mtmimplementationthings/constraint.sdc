# ####################################################################

#  Created by Genus(TM) Synthesis Solution 21.14-s082_1 on Wed Nov 04 02:20:13 IST 2026

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1000fF
set_units -time 1000ps

# Set the current design
current_design mxdotp_unit

create_clock -name "clk" -period 5.0 -waveform {0.0 2.5} [get_ports clk]
set_clock_gating_check -setup 0.0 
set_wire_load_mode "enclosed"
