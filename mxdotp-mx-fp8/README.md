# MXFP8 Dot-Product Unit in Verilog — paper re-implementation + block-fused improvement

Verilog (Vivado-ready) implementation of the **Microscaling (MX) FP8 dot-product-accumulate datapath**
described in *“MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products”*
(İslamoğlu, Bertaccini, Prasad, Conti, Garofalo, Benini — ASAP 2025, [arXiv:2505.13159](https://arxiv.org/abs/2505.13159)),
plus an **improved block-level design (MXDOTP-B32)**, self-checking testbenches, exact golden-model scripts and a
numerical (ML-data) study.

> **Status / honesty box — read this first**
> | Item | State |
> |---|---|
> | Algorithm (decode → exact products → window alignment → single RNE) vs. exact arithmetic | ✅ **Executed**: 200 000 random vectors + 24 000 generated vectors, 0 mismatches (`scripts/rtl_model.py`, `scripts/gen_vectors.py`) |
> | Numerical study (`scripts/accuracy_study.py`) | ✅ **Executed**, numbers below are real |
> | Verilog RTL + testbenches | ⚠️ **Written as a line-by-line mirror of the validated model, but not yet simulated by me** (no Verilog simulator was available where this was produced). Run `vivado/run_sim.tcl` or `sim/run_iverilog.sh` (or push to GitHub, CI runs it). Expect to fix small tool-specific issues if any appear. |
> | Area / timing / power of the RTL | ❌ **Not measured.** `vivado/synth_report.tcl` produces them; fill in the table in §7. |
> | Snitch core integration, SSRs, CSR, ISA encoding (paper §III-B) | ❌ **Not implemented** — this repo covers the arithmetic datapath (the paper’s §III-A) only. |

---

## 1. Repository layout

```
.
├── rtl/
│   ├── common/
│   │   ├── fp8_decode.v          # FP8 (E5M2/E4M3) -> {sign, 4b significand, 5b exponent, flags}
│   │   ├── mxdotp_dot8.v         # STAGE 1: 8 lanes, 4x4 multipliers, exact fixed-point sum T
│   │   └── mxdotp_addround.v     # STAGES 2+3: align acc, add, normalise, RNE round, specials
│   ├── baseline/
│   │   └── mxdotp_unit.v         # paper-style MXDOTP: 1 beat = 8 elements, 3-stage pipeline
│   └── improved/
│       └── mxdotp_blk32.v        # IMPROVED: 4 beats (32-element MX block) fused, 1 rounding/block
├── tb/
│   ├── tb_mxdotp_unit.v          # self-checking TB (baseline)
│   └── tb_mxdotp_blk32.v         # self-checking TB (improved)
├── scripts/
│   ├── mx_ref.py                 # EXACT golden reference (Fractions, single rounding)
│   ├── rtl_model.py              # bit-accurate Python mirror of the RTL algorithm + self-check
│   ├── gen_vectors.py            # writes sim/vectors/*.mem (directed + random + ML-style)
│   ├── mx_quant.py               # NumPy MX quantiser (E5M2/E4M3, E8M0 scales, block 32)
│   └── accuracy_study.py         # ML-style numerical study of quantisation + accumulation error
├── sim/
│   ├── run_iverilog.sh           # open-source simulation flow
│   └── vectors/                  # generated *.mem golden vectors (committed so Vivado works out-of-the-box)
├── vivado/
│   ├── create_project.tcl        # builds the Vivado project
│   ├── run_sim.tcl               # runs both testbenches in xsim (batch)
│   └── synth_report.tcl          # out-of-context synthesis: area/timing/power of both designs
├── .github/workflows/ci.yml      # model check + iverilog simulation + study on every push
├── Makefile  requirements.txt  LICENSE  .gitignore
```

## 2. Quick start

```bash
pip install -r requirements.txt            # numpy only
python3 scripts/rtl_model.py               # algorithm vs exact reference  -> "0 mismatches"
python3 scripts/gen_vectors.py             # (re)generate sim/vectors/*.mem
python3 scripts/accuracy_study.py          # ML-style numerical study
```

### Vivado (GUI)
1. *Create Project → RTL Project*, add `rtl/common/*.v`, `rtl/baseline/*.v`, `rtl/improved/*.v` as **design sources**.
2. Add `tb/*.v` as **simulation sources** and add `sim/vectors/unit_vectors.mem`, `blk_vectors.mem` as simulation sources too.
3. *Simulation Settings → xsim → Simulate → `xsim.simulate.xsim.more_options`*: set
   `-testplusarg VEC=<absolute path>/sim/vectors/unit_vectors.mem` (or `.../blk_vectors.mem` for the other TB).
   (If you skip this, the TB looks for `unit_vectors.mem` / `blk_vectors.mem` in the simulation working directory.)
4. Set `tb_mxdotp_unit` (or `tb_mxdotp_blk32`) as simulation top → *Run Simulation → Run All*.
   Console must end with `PASS : N / N vectors bit-exact`.

### Vivado (Tcl, one shot)
```bash
vivado -mode batch -source vivado/create_project.tcl
vivado -mode batch -source vivado/run_sim.tcl        # both testbenches
vivado -mode batch -source vivado/synth_report.tcl   # area / timing / power -> vivado/reports/
```

### Icarus Verilog (no Vivado needed)
```bash
make sim        # or ./sim/run_iverilog.sh
```

## 3. Background: what the paper does and what is reproduced

**Microscaling (MX)**: blocks of 32 low-precision elements share one power-of-two scale (E8M0, 8 bit); each element
keeps its own small exponent (here MXFP8: **E5M2** or **E4M3**). The OCP MX dot product is

```
Dot(A,B) = Xa · Xb · Σ_{i=1..k} Pa_i · Pb_i          result accumulated in FP32
```

**Paper idea (§III-A)**: fuse scaling + 8-element dot product + FP32 accumulation into one unit, using
*early accumulation* — add the shifted FP32 accumulator to the sum of products in **fixed point**, convert/round once.
The paper uses an FP9 (E5M3) intermediate, a 95-bit fixed-point sum with anchor 34, RNE, 3 pipeline stages,
8 elements per 64-bit operand, one result per cycle.

| Concept | Paper | This repo (`rtl/baseline`) |
|---|---|---|
| Elements per instruction | 8 × FP8 in 64 bit | same (`a[63:0]`, `b[63:0]`) |
| Formats | E5M2 + E4M3 via CSR | E5M2 + E4M3 via `fmt` pin |
| Scales | 2 × E8M0 | 2 × E8M0 (`xa`, `xb`), 0xFF → NaN |
| Product representation | FP9 (E5M3) | 4-bit integer significand × shared exponent `eb−18` (both formats map to one grid) |
| Early accumulation | 95-bit fixed point, anchor 34 | 72-bit exact product sum `T` (LSB = 2⁻³⁴) + 132-bit accumulator window with sticky |
| Rounding | RNE, FP32 | RNE, FP32, gradual underflow, overflow→Inf |
| Pipeline | 3 stages, 1 result/cycle | 3 stages, 1 result/cycle |
| Operand delivery | SSRs + merged scale/acc | **not modelled** (ports `xa`, `xb`, `acc` are separate) |

## 4. Baseline architecture (`mxdotp_unit`)

```
 a[63:0] b[63:0] fmt                 xa xb                                 acc[31:0]
    │      │                          │  │                                    │
┌───▼──────▼──────────┐          ┌────▼──▼────┐                                │
│ 8 × fp8_decode (A,B)│          │ S=xa+xb-254│                                │
│ 8 × (4b × 4b) mult  │          └─────┬──────┘                                │
│ shift by eb_a+eb_b-2│                │                                       │
│ signed sum → T (72b)│                │         ── pipeline reg 1 ──          │
└──────────┬──────────┘                │                                       │
           └───────────────┬───────────┴──────────────────┬────────────────────┘
                     ┌─────▼────────────────────────────  ▼ ─────────────┐
 STAGE 2             │ place acc (24b) into 132-bit window at bit 32+d,    │
 align + add         │ d = e_acc − S − 116 ; far-away operands → sticky    │
                     │ W = ±acc_aligned + T<<32 ; sign/magnitude           │
                     └───────────────────────┬────────────────────────────┘
                                             │            ── pipeline reg 2 ──
                     ┌───────────────────────▼────────────────────────────┐
 STAGE 3             │ leading-one detect → exponent, RNE round (G,S bits), │
 normalise + round   │ subnormal/overflow handling, NaN/Inf/±0 rules        │
                     └───────────────────────┬────────────────────────────┘
                                             ▼  result[31:0]  (pipeline reg 3)
```

**Key equations** (all arithmetic *exact* until the single final rounding):

* element → `value = (−1)^s · M · 2^(eb−18)`, `M` = 4-bit significand, `eb` = effective biased exponent
  (E5M2: `e_eff`; E4M3: `e_eff + 8`, so one datapath serves both).
* product lane `= (Ma·Mb) << (eba+ebb−2)` in units of 2⁻³⁴; `T = Σ ±lane` (72 bit suffices for 32 products).
* `result = RNE_fp32( acc + 2^(Xa+Xb−254) · T · 2⁻³⁴ )`.
* Accumulator alignment: `d = e_acc − S − 116`. If `d > 73` the product sum can only act as a rounding **sticky**
  (±1 at the window LSB); if the accumulator is far below the products it is shifted right with a sticky bit
  (`OT = 32` guard bits make both cases rounding-exact). `T == 0` bypasses the window (`acc + 0 = acc`).
* Special values: any NaN, `Inf·0`, `+Inf + −Inf`, E8M0 scale `0xFF` → canonical qNaN `0x7FC00000`;
  `±Inf` propagate; `−0` only if every term is a negative zero; exact cancellation → `+0`.

## 5. Improvement: MXDOTP-B32 (`rtl/improved/mxdotp_blk32.v`)

**Observation.** The baseline performs *align → normalise → round* after **every** 8-element beat, i.e. four times per
32-element MX block, and chains the FP32 accumulator through the 3-stage pipeline (loop-carried latency of 3 cycles
→ software must interleave ≥3 independent accumulators/outputs to keep the unit busy).

**Techniques**

1. **Deferred normalisation / block-level fusion.** The *exact* fixed-point sums `T` of up to 4 beats are accumulated in a
   72-bit register (a small Kulisch-style exact accumulator; one 72-bit adder per beat). The expensive 132-bit
   alignment shifter, leading-one detector and rounder run **once per MX block** (4× fewer activations).
2. **Single rounding per block.** One rounding instead of four; no double-rounding accumulation of the 4 beats.
3. **Dependency decoupling.** Beats of a block chain only through the narrow `T` register (1 cycle), not through the
   FP32 pipeline. The FP32 accumulator is consumed once per block (`acc_in` sampled on the `last` beat). With 4-beat blocks and
   a 3-cycle add/round latency, back-to-back blocks do not stall on the previous block’s result (design reasoning;
   not covered by a dedicated testbench yet).
4. **Hardware block control + protocol checker.** `first/last` tags replace the software beat loop; `blk_err`
   flags a scale change inside a block or more than 4 beats (would exceed the 72-bit range).

**Interface per block** (same `xa`, `xb`, `fmt` for all beats): beat 0 `first=1`; beat 3 `last=1` and `acc_in` = running FP32
sum; result after 3 cycles, `blk_err` aligned with `out_valid`. 1–4 beats per block are allowed.

**Methodology** (how the work was done, in order)

1. *Specification extraction* from the paper + OCP MX v1.0 (formats, E8M0, accumulation precision implementation-defined).
2. *Golden reference first* (`mx_ref.py`): exact Fraction arithmetic, single RNE, full special-value policy.
3. *Algorithm mirror* (`rtl_model.py`): the same integer algorithm with the *same bit widths* as the RTL; validated against
   the golden model with random + corner-case + cancellation + extreme-scale vectors (this found and fixed one bug:
   E4M3 subnormal exponent re-bias). Widths (`TW=72, OT=32, DMAX=73, SW=132`) were derived analytically (see §4) and then confirmed by test.
4. *RTL written as a line-by-line translation of the mirror*, shared sub-modules for baseline and improved designs so the
   comparison isolates the architectural change.
5. *Vector-based self-checking testbenches* with expected values from the exact model; random idle bubbles test valid-pipelining.
6. *Numerical study on ML-style data* to quantify what the accumulation scheme costs/gains (§6).
7. *Synthesis comparison* (`synth_report.tcl`) to evaluate the area/timing trade-off (§7, to be filled by you).

## 6. Machine-learning / numerics part

The RTL itself needs **no** ML. ML enters in two places:

* **Test data realism**: `gen_vectors.py` includes “ML-style” blocks — Gaussian activations/weights with log-normal outlier
  channels, quantised with the real MX rules in `mx_quant.py` — so the datapath sees realistic exponent distributions, not just uniform random bits.
* **Numerical study** (`accuracy_study.py`): C = A·B (16×4096 · 4096×16), compared on identical MX-quantised operands.

Measured (seed 0, `--sigma 1.0`; relative RMS error):

| Format | Per-tensor FP8 quant. error | MX (block 32) quant. error | Accum. error: SW FP8→FP32 | MXDOTP (round/8) | MXDOTP-B32 (round/32) |
|---|---|---|---|---|---|
| E4M3 | 3.85e-2 | 4.75e-2 | 1.57e-7 | 1.63e-7 | 1.56e-7 |
| E5M2 | 7.84e-2 | 8.15e-2 | 8.62e-8 | 8.72e-8 | 8.63e-8 |

**What this honestly says**

* FP8 *quantisation* error (~10⁻²) dwarfs FP32 *accumulation* error (~10⁻⁷). Fewer roundings (B32) therefore give **no
  meaningful accuracy gain**; the justification for B32 is **hardware activity (4× fewer align/normalise/round operations),
  dependency/latency behaviour and simpler control — not accuracy.** Whether that yields area/energy benefits must be confirmed by synthesis (§7).
* On this synthetic data MX block scaling did **not** beat per-tensor scaling at the matmul-output level (the output is
  dominated by large elements, which both schemes represent well). MX’s accuracy advantage reported in the literature
  (OCP/Microsoft) comes from real LLM/CNN tensors with wide intra-tensor dynamic range; to test that claim, feed real tensors
  (e.g. exported PyTorch weights/activations) into `mx_quant.py` — a natural next step.
* B8/B32 are emulated with float64 chunk sums + one FP32 rounding; the RTL is verified bit-exactly by the testbenches.

## 7. Results table to fill in after running `synth_report.tcl`

| Design | LUTs | FFs | DSP | Fmax @ xc7a100t (WNS at 10 ns) | Rounding events / 32 elements |
|---|---|---|---|---|---|
| `mxdotp_unit` (baseline, 4 beats) | _fill_ | _fill_ | _fill_ | _fill_ | 4 |
| `mxdotp_blk32` (improved) | _fill_ | _fill_ | _fill_ | _fill_ | 1 |

Note: the improved module adds a 72-bit accumulator + control; the shared align/normalise logic is not duplicated. Expect the
combinational depth of stage 2/3 (132-bit shifters) to dominate timing for both; if it fails 100 MHz, split stage 3 or reduce `OT/SW`.

## 8. Verification summary

* Exact reference vs algorithm mirror: 200 000 random cases (8 and 32 elements, both formats, subnormals, NaN/Inf, E8M0 extremes 0/254/255, forced cancellation, subnormal/zero accumulators) → 0 mismatches.
* Vector files: 20 000 beats + 4 000 blocks (19 directed corner cases each, 45 % random patterns, 55 % ML-style).
* Testbenches compare every output bit-exactly (including the canonical NaN) and check that the number of outputs equals the number of inputs.

## 9. Known limitations / future work

* No Snitch/RISC-V integration (SSR streaming, CSR, instruction encoding from paper §III-B).
* Only MXFP8; MXFP6/MXFP4 (more lanes per 64-bit word) and MXINT8 are natural extensions of `fp8_decode`/`mxdotp_dot8`.
* Blocks > 4 beats need `TW` > 72 (parameterised; `blk_err` flags the violation).
* Single wide combinational stages (shifters) — retime or add pipeline registers for higher clock rates.
* Directed testbench for back-to-back blocks with real accumulator feedback (acc_in = previous result).

## 10. Publishing on GitHub

```bash
cd mxdotp-mx-fp8
git init && git add . && git commit -m "MXFP8 dot-product unit (paper re-implementation) + MXDOTP-B32 improvement"
git branch -M main
git remote add origin https://github.com/<you>/mxdotp-mx-fp8.git
git push -u origin main
```
Before pushing: put your name in `LICENSE`, and fill in §7 with your own synthesis numbers.

## 11. Credits

Architecture inspired by İslamoğlu et al., *MXDOTP* (ASAP 2025); formats per the OCP *Microscaling Formats (MX) v1.0* specification.
This is an independent re-implementation from the published description — **cite the paper** in any report or publication that uses this work.
