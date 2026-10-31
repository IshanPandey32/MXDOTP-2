#!/usr/bin/env bash
# Open-source alternative to Vivado:  ./sim/run_iverilog.sh      (needs: iverilog, python3)
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/gen_vectors.py
mkdir -p build
iverilog -g2012 -o build/unit.vvp rtl/common/*.v rtl/baseline/*.v tb/tb_mxdotp_unit.v
iverilog -g2012 -o build/blk.vvp  rtl/common/*.v rtl/improved/*.v tb/tb_mxdotp_blk32.v
vvp build/unit.vvp +VEC=sim/vectors/unit_vectors.mem | tee build/unit.log
vvp build/blk.vvp  +VEC=sim/vectors/blk_vectors.mem  | tee build/blk.log
grep -q "^PASS" build/unit.log && grep -q "^PASS" build/blk.log && echo "ALL TESTS PASSED"
