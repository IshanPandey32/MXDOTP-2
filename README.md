# MXDOTP-2: Modular Block-32 OCP MXFP8 Dot-Product Accelerator (45nm CMOS)

**MXDOTP-2** is a standalone Verilog RTL implementation and 45nm CMOS ASIC evaluation of an **OCP Microscaling Floating-Point (MXFP8)** dot-product accelerator. It re-implements the modular arithmetic datapath from İslamoğlu et al. (*"MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products"*, ASAP 2025, arXiv:2505.13159) as a clean baseline (`mxdotp_unit`), and introduces a **block-32 fused architecture (`mxdotp_blk32` / MXDOTP-B32)** that defers FP32 normalization and rounding to once per 32-element MX block.

## 1. Project Status & Verification Scorecard

| Component | Status | Verification & Tooling Summary |
| :--- | :--- | :--- |
| **Golden Reference Model (`scripts/rtl_model.py`, `gen_vectors.py`)** | Completed | 200,000 random test cases + 24,000 generated vectors (20,000 beats + 4,000 blocks), **0 mismatches** |
| **Numerical Accuracy Study (`scripts/accuracy_study.py`)** | Completed | Relative RMS error evaluated for E4M3 and E5M2 across per-tensor FP8, MX block-32, SW FP32, Baseline, and B32 |
| **Baseline RTL (`mxdotp_unit.v`) — Cadence `ncsim`** | Passed | **12 / 12** golden vectors bit-exact (`mtmimplementationthings/ncsim.log`, 275 ns) |
| **Innovation RTL (`mxdotp_blk32.v`) — Cadence `ncsim`** | Passed | **2 / 2** full 32-element blocks bit-exact (`mtm_innoinnovation/ncsim.log`, 235 ns) |
| **45nm ASIC Synthesis (Cadence Genus 21.14)** | Completed | Synthesized at `slow_vdd1v0 1.0` (`PVT_0P9V_125C`), 5.0 ns (200 MHz) target clock for both designs |
| **Physical Implementation (Cadence Innovus)** | Completed | Place-and-route logs and chip layout captures archived in `mtmimplementationthings/` and `mtm_innoinnovation/` |
| **Snitch Core / SSR / CSR / RISC-V ISA Integration** | Out of Scope | Standalone arithmetic datapath only (paper §III-A) |

## 2. Repository Structure

| Path | Description |
| :--- | :--- |
| `mxdotp-mx-fp8/rtl/common/` | Shared arithmetic primitives: `fp8_decode.v`, `mxdotp_dot8.v`, `mxdotp_addround.v` |
| `mxdotp-mx-fp8/rtl/baseline/` | Baseline 8-element/beat datapath (`mxdotp_unit.v`) |
| `mxdotp-mx-fp8/rtl/improved/` | Proposed block-32 fused datapath (`mxdotp_blk32.v`) |
| `mxdotp-mx-fp8/tb/` | Verilog testbenches for baseline and block-32 designs |
| `mxdotp-mx-fp8/scripts/` | Bit-exact Python golden model (`rtl_model.py`), vector generator (`gen_vectors.py`), MX quantizer (`mx_quant.py`), and accuracy study (`accuracy_study.py`) |
| `mxdotp-mx-fp8/sim/` & `vivado/` | Icarus Verilog runner (`run_iverilog.sh`) and Vivado Tcl synthesis script (`synth_report.tcl`) |
| `mtmimplementationthings/` | Baseline 45nm ASIC artifacts: `mxdotp_unit.v`, `constraint.sdc`, `area.rpt`, `power.rpt`, `timing.rpt`, `ncsim.log`, `innovus.log`, and 10 visual captures |
| `mtm_innoinnovation/` | Innovation (`MXDOTP-B32`) 45nm ASIC artifacts: `mxdotp_blk32.v`, `constraint.sdc`, `area.rpt`, `power.rpt`, `timing.rpt`, `ncsim.log`, `innovus.log`, and 9 visual captures |

## 3. Architecture & Block-32 Fused Innovation

### 3.1 OCP MXFP8 Dot-Product Formulation

Under the **OCP Microscaling Formats (MX) v1.0** specification, a 32-element block shares a single 8-bit E8M0 scale factor (`Xa`, `Xb`) across 32 FP8 elements (`Pa,i`, `Pb,i` in E4M3 or E5M2):

$$\text{Dot}(A, B) = X_a \cdot X_b \cdot \sum_{i=1}^{k} P_{a,i} \cdot P_{b,i}$$

### 3.2 Shared RTL Core (`mxdotp-mx-fp8/rtl/common/`)

| Module | Pipeline Role | Key Architectural Parameters |
| :--- | :--- | :--- |
| `fp8_decode.v` | Unified FP8 decoder | Normalizes both E5M2 and E4M3 into a unified 4-bit significand and 5-bit effective exponent |
| `mxdotp_dot8.v` | Stage 1 exact 8-way dot product | Computes exact 8-element integer products and aligns into a 72-bit two's-complement fixed-point sum (`TW = 72`) |
| `mxdotp_addround.v` | Stages 2–3 FP32 accumulate & round | 132-bit fixed-point alignment (`SW = 132`, `DMAX = 73`), addition with FP32 accumulator (`OT = 32`), leading-one detection (LOD), and Round-to-Nearest-Even (RNE) |

### 3.3 Baseline (`mxdotp_unit`) vs. Innovation (`mxdotp_blk32` / MXDOTP-B32)

| Architectural Feature | Baseline: `mxdotp_unit` | Innovation: `mxdotp_blk32` (`MXDOTP-B32`) |
| :--- | :--- | :--- |
| **Throughput per Beat** | 8 MXFP8 elements / beat | 8 MXFP8 elements / beat |
| **MX Block Granularity** | Beat-level (4 independent beats per 32-element block) | Block-level fused (4 beats accumulated per 32-element MX block) |
| **Rounding & Normalization Events / 32 Elements** | **4** (`mxdotp_addround` fires every beat) | **1** (`mxdotp_addround` fires once on `last` beat — **4x reduction**) |
| **Intermediate Accumulation** | Immediate FP32 alignment + RNE rounding every 8 elements | Exact 72-bit running fixed-point accumulator (`t_run[71:0]`) across 4 beats |
| **Block Framing & Protocol Safety** | Beat-level valid handshake | Explicit `first` / `last` beat tags with built-in `blk_err` protocol violation detection |
| **Primary Hardware Advantage** | Lower combinational depth & smaller register footprint | 4x fewer FP32 align/normalize/round activations; enables clock gating on `u_addround` |

## 4. Verification & Numerical Accuracy Results

### 4.1 Functional Verification Summary

| Verification Stage | Design Under Test | Vector / Test Scope | Result | Source Log |
| :--- | :--- | :--- | :--- | :--- |
| **Python Golden Model** | Both (`rtl_model.py`) | 200,000 random cases | **0 mismatches** | `mxdotp-mx-fp8/scripts/rtl_model.py` |
| **Generated Vector Suite** | Both (`gen_vectors.py`) | 20,000 beats + 4,000 blocks (24,000 total) | **0 mismatches** | `mxdotp-mx-fp8/scripts/gen_vectors.py` |
| **Cadence `ncsim(64) 15.20-s086`** | Baseline (`mxdotp_unit`) | 12 built-in golden vectors (`tb_mxdotp_unit.v`) | **PASS: 12 / 12 bit-exact** (275 ns) | [ncsim.log](https://www.google.com/search?q=mtmimplementationthings/ncsim.log) |
| **Cadence `ncsim(64) 15.20-s086`** | Innovation (`mxdotp_blk32`) | 2 built-in 32-element blocks (`tb_mxdotp_blk32.v`) | **PASS: 2 / 2 blocks bit-exact** (235 ns) | [ncsim.log](https://www.google.com/search?q=mtm_innoinnovation/ncsim.log) |

### 4.2 Numerical Accuracy Study (`accuracy_study.py`, seed 0, `--sigma 1.0`)

Relative RMS error comparing quantization error against accumulation error:

| Format | Per-Tensor FP8 Quant. Error | MX (Block 32) Quant. Error | Accum. Error: SW FP8 → FP32 | Baseline: MXDOTP (Round / 8) | Innovation: MXDOTP-B32 (Round / 32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E4M3** | `3.85e-2` | `4.75e-2` | `1.57e-7` | `1.63e-7` | **`1.56e-7`** |
| **E5M2** | `7.84e-2` | `8.15e-2` | `8.62e-8` | `8.72e-8` | **`8.63e-8`** |

**Presentation Takeaway:** FP8 quantization noise ($\sim 10^{-2}$) dominates FP32 accumulation rounding error ($\sim 10^{-7}$) by five orders of magnitude. While `MXDOTP-B32` slightly improves accumulation accuracy (matching software FP32 to $1.56\times 10^{-7}$ for E4M3 and $8.63\times 10^{-8}$ for E5M2), its primary engineering value is **architectural efficiency**: reducing expensive FP32 alignment, LOD, and RNE rounding operations by **4x per MX block**.

## 5. 45nm CMOS ASIC Synthesis Results (Cadence Genus 21.14)

Both designs were synthesized using **Cadence Genus 21.14** targeting the 45nm standard-cell library at the **`slow_vdd1v0 1.0` (`PVT_0P9V_125C`)** worst-case corner with a **5.0 ns (200 MHz)** clock constraint.

### 5.1 Area, Power, and Timing Comparison

| Synthesis Metric | Baseline: `mxdotp_unit` | Innovation: `mxdotp_blk32` | Delta (B32 vs. Baseline) |
| :--- | :---: | :---: | :---: |
| **Target Clock Period / Freq** | 5,000 ps (200 MHz) | 5,000 ps (200 MHz) | Same constraint |
| **Total Standard Cell Count** | 9,438 | 13,731 | +4,293 (+45.5%) |
| **Total Cell Area (µm²)** | 18,926.605 | 26,033.160 | +7,106.555 (+37.5%) |
| **Net Area** | 0.000 | 0.000 | — |
| **Key Submodule Breakdown** | `u_dot8` (`mxdotp_dot8_TW72`): 3,176 cells, 6,895.404 µm² | `u_addround` (`mxdotp_addround_TW72`): 9,018 cells, 15,310.092 µm² | `u_addround` dominates area |
| **Leakage Power (W)** | `4.07577e-07` (0.03%) | `6.20459e-07` (0.03%) | +52.2% |
| **Internal Power (W)** | `1.05039e-03` (73.68%) | `1.57733e-03` (69.80%) | +50.2% |
| **Switching Power (W)** | `3.74734e-04` (26.29%) | `6.81879e-04` (30.17%) | +82.0% |
| **Total Power (mW)** | **1.426 mW** (`1.42553e-03 W`) | **2.260 mW** (`2.25983e-03 W`) | +0.834 mW (+58.5%, vectorless) |
| **Worst Negative Slack (WNS)** | **0 ps (MET)** | **-1,324 ps (VIOLATION)** | -1,324 ps |
| **Data Arrival + Setup Time** | 4,923 ps + 77 ps = 5,000 ps | 6,205 ps + 120 ps = 6,324 ps | Effective Fmax ≈ **158.1 MHz** |
| **Critical Path** | `u_addround_s2_mag_reg[48]/CK` → `u_addround_result_reg[22]/D` | `u_addround/s1_t_mag_reg[36]/CK` → `u_addround/result_reg[30]/D` | Bottleneck inside `u_addround` |
| **Rounding Events / 32 Elements** | **4** | **1** | **-75% (4x fewer)** |

### 5.2 Key Engineering Trade-Offs (Presentation Talking Points)

1. **Why vectorless power is higher in `mxdotp_blk32`:** Cadence Genus default power analysis uses uniform vectorless switching activity without clock gating on `u_addround`. Because `u_addround` accounts for 9,018 of 13,731 cells (58.8% of total area) yet only needs to compute on 1 out of every 4 cycles (`last == 1`), adding operand isolation or clock gating on `u_addround` directly unlocks the 4x activity reduction in real block-32 workloads.
2. **Why `mxdotp_blk32` misses 200 MHz by 1,324 ps:** The critical path runs inside `u_addround` (`s1_t_mag_reg[36]` → `result_reg[30]`, 6.324 ns total, corresponding to ~158.1 MHz at worst-case 0.9V / 125°C). Adding a pipeline register stage across the wide 132-bit shifter/LOD boundary in `mxdotp_addround.v` will close timing at 200 MHz+.
3. **Area overhead:** The +37.5% cell area in `mxdotp_blk32` stems from the 72-bit running accumulator (`t_run[71:0]`), block-framing control/error logic (`first`, `last`, `blk_err`), and boundary preservation around `u_addround`.

## 6. Side-by-Side Visual Quick Reference (For Live Presentation)

Use this table during presentations or reviews to compare Baseline (`mxdotp_unit`) and Innovation (`mxdotp_blk32`) results side by side.

| Stage | Baseline: `mxdotp_unit` (`mtmimplementationthings/`) | Innovation: `mxdotp_blk32` (`mtm_innoinnovation/`) |
| :--- | :---: | :---: |
| **Simulation Waveform** |  |  |
| **Synthesized Schematic** |  |  |
| **Gate Count / Area** |  |  |
| **Power Report** |  |  |
| **Timing Report** |  |  |

## 7. Baseline Implementation Gallery (`mtmimplementationthings/`)

**Raw Reports & Design Files:** [mxdotp\_unit.v](https://www.google.com/search?q=mtmimplementationthings/mxdotp_unit.v) | [constraint.sdc](https://www.google.com/search?q=mtmimplementationthings/constraint.sdc) | [area.rpt](https://www.google.com/search?q=mtmimplementationthings/area.rpt) | [power.rpt](https://www.google.com/search?q=mtmimplementationthings/power.rpt) | [timing.rpt](https://www.google.com/search?q=mtmimplementationthings/timing.rpt) | [ncsim.log](https://www.google.com/search?q=mtmimplementationthings/ncsim.log) | [innovus.log](https://www.google.com/search?q=mtmimplementationthings/innovus.log)

### 7.1 Functional Simulation Waveform (`ncsim`)


### 7.2 Synthesized RTL Schematic (Full & Zoomed)




### 7.3 Gate Count & Mapped Standard-Cell Netlist




### 7.4 Power, Timing & Critical Path Analysis






### 7.5 Additional Baseline Implementation Captures




## 8. Innovation (`MXDOTP-B32`) Gallery (`mtm_innoinnovation/`)

**Raw Reports & Design Files:** [mxdotp\_blk32.v](https://www.google.com/search?q=mtm_innoinnovation/mxdotp_blk32.v) | [constraint.sdc](https://www.google.com/search?q=mtm_innoinnovation/constraint.sdc) | [area.rpt](https://www.google.com/search?q=mtm_innoinnovation/area.rpt) | [power.rpt](https://www.google.com/search?q=mtm_innoinnovation/power.rpt) | [timing.rpt](https://www.google.com/search?q=mtm_innoinnovation/timing.rpt) | [ncsim.log](https://www.google.com/search?q=mtm_innoinnovation/ncsim.log) | [innovus.log](https://www.google.com/search?q=mtm_innoinnovation/innovus.log)

### 8.1 Functional Simulation Waveform (`ncsim`)


### 8.2 Synthesized RTL Schematic (Full & Zoomed)




### 8.3 Gate Count & Mapped Standard-Cell Netlist




### 8.4 Power & Timing Analysis




### 8.5 Physical Chip Layout (Cadence Innovus — Full & Zoomed)




## 9. Quick Start & Reproduction

All Python scripts, RTL sources, and open-source simulation flows reside inside `mxdotp-mx-fp8/`:

```bash
cd mxdotp-mx-fp8

# 1. Run the bit-exact Python golden model self-check (200,000 random cases)
python3 scripts/rtl_model.py

# 2. Generate 20,000 beat vectors + 4,000 block-32 vectors
python3 scripts/gen_vectors.py

# 3. Reproduce the E4M3 / E5M2 numerical accuracy study
python3 scripts/accuracy_study.py

# 4. Run open-source RTL simulation with Icarus Verilog
bash sim/run_iverilog.sh
```

For FPGA synthesis (Vivado) or ASIC reproduction, see `mxdotp-mx-fp8/vivado/synth_report.tcl` and the Cadence constraints/reports in `mtmimplementationthings/` and `mtm_innoinnovation/`.

## 10. Known Limitations & Future Work

- **Clock Gating / Operand Isolation:** Gate `u_addround` in `mxdotp_blk32` during beats 0–2 (`last == 0`) and run SAIF/VCD activity-annotated power analysis to measure dynamic power savings from the 4x rounding reduction.
- **Timing Closure at 200 MHz+:** Pipeline the wide 132-bit combinational shifter/LOD path inside `mxdotp_addround.v` to eliminate the `-1,324 ps` violation on `mxdotp_blk32`.
- **Larger Block Sizes:** Blocks exceeding 4 beats (>32 elements) require widening the fixed-point accumulator beyond `TW = 72`.
- **Format & System Expansion:** Extend beyond MXFP8 (E4M3/E5M2) to MXFP6/MXFP4 and integrate with a RISC-V Snitch/SSR host pipeline.

## 11. Credits & References

- **Baseline Architecture:** İslamoğlu et al., *"MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products"*, IEEE ASAP 2025 (arXiv:2505.13159).
- **Format Standard:** Open Compute Project (OCP), *Microscaling Formats (MX) Specification v1.0*.


Contributors: Divyansh Rawat and Ishan Pandey
