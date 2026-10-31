"""
gen_vectors.py - golden test-vector generator for the Vivado testbenches.

  python3 scripts/gen_vectors.py [--n-unit 20000] [--n-blk 4000] [--seed 1] [--out sim/vectors]

Outputs  unit_vectors.mem   (tb_mxdotp_unit : 8-element beats)
         blk_vectors.mem    (tb_mxdotp_blk32 : 32-element MX blocks = 4 beats)
Expected values come from the EXACT Fraction reference (mx_ref.py), and every vector is
also cross-checked against the bit-accurate algorithm mirror (rtl_model.py).

Vector classes:  directed corner cases | random bit patterns (specials, subnormals, extreme
scales, cancellation) | "ML-style" blocks: Gaussian tensors with outlier channels, quantised
with real MX rules (mx_quant.py) so the datapath sees realistic exponent distributions.
"""
import argparse
import os
import random
import numpy as np

from mx_ref import mxdotp_ref
from rtl_model import rand_case, mxdotp_hw
import mx_quant as mq


def pack_words(lst):
    w = 0
    for i, x in enumerate(lst):
        w |= (x & 0xFF) << (8 * i)
    return w


def unit_line(a, b, xa, xb, acc, fmt, exp):
    v = pack_words(a)
    v = (v << 64) | pack_words(b)
    v = (v << 8) | xa
    v = (v << 8) | xb
    v = (v << 32) | acc
    v = (v << 4) | fmt
    v = (v << 32) | exp
    return f"{v:053x}"


def blk_line(a, b, xa, xb, acc, fmt, exp):
    A = sum(pack_words(a[8 * j:8 * j + 8]) << (64 * j) for j in range(4))
    B = sum(pack_words(b[8 * j:8 * j + 8]) << (64 * j) for j in range(4))
    v = (A << 256) | B
    v = (v << 8) | xa
    v = (v << 8) | xb
    v = (v << 32) | acc
    v = (v << 4) | fmt
    v = (v << 32) | exp
    return f"{v:0149x}"


def pad(lst, n):
    return (lst + [0] * n)[:n]


def directed(n):
    """Hand-written corner cases (lists of length n, n = 8 or 32)."""
    one5, one4 = 0x3C, 0x38                     # 1.0 in E5M2 / E4M3
    c = []
    c.append((pad([one5] * 8, n), pad([one5] * 8, n), 127, 127, 0, 0))
    c.append((pad([one4] * 8, n), pad([one4] * 8, n), 127, 127, 0x3F800000, 1))
    c.append((pad([one4] * 8, n), pad([0xB8] * 8, n), 127, 127, 0x41000000, 1))       # cancel to 0
    c.append((pad([0x7D] + [one5] * 7, n), pad([one5] * 8, n), 127, 127, 0, 0))        # NaN (E5M2)
    c.append((pad([0x7C] + [one5] * 7, n), pad([one5] * 8, n), 127, 127, 0, 0))        # +Inf
    c.append((pad([0x7C, 0xFC] + [0] * 6, n), pad([one5] * 8, n), 127, 127, 0, 0))     # +Inf + -Inf
    c.append((pad([0x7C] + [0] * 7, n), pad([0x00] * 8, n), 127, 127, 0, 0))           # Inf*0
    c.append((pad([0x7F] * 8, n), pad([one4] * 8, n), 127, 127, 0, 1))                 # NaN (E4M3)
    c.append((pad([one4] * 8, n), pad([one4] * 8, n), 255, 127, 0, 1))                 # scale NaN
    c.append((pad([one4] * 8, n), pad([one4] * 8, n), 127, 127, 0x7FC00001, 1))        # acc NaN
    c.append((pad([one4] * 8, n), pad([one4] * 8, n), 127, 127, 0xFF800000, 1))        # acc -Inf
    c.append((pad([0x80] * 8, n), pad([0x00] * 8, n), 127, 127, 0x80000000, 0))        # -0 + -0
    c.append((pad([0x80] * 8, n), pad([0x00] * 8, n), 127, 127, 0x00000000, 0))        # -0 + +0
    c.append((pad([0x7B] * 8, n), pad([0x7B] * 8, n), 254, 254, 0, 0))                 # overflow -> Inf
    c.append((pad([0x01] * 8, n), pad([0x01] * 8, n), 0, 0, 0x00000001, 0))            # deep underflow
    c.append((pad([0x01] * 8, n), pad([0x01] * 8, n), 100, 20, 0x00000001, 0))         # subnormal result
    c.append((pad([0x38] * 8, n), pad([0x38] * 8, n), 0, 0, 0x7F7FFFFF, 1))            # acc dominates
    c.append((pad([0x38] * 8, n), pad([0x38] * 8, n), 254, 254, 0x00800000, 1))        # product dominates
    c.append((pad([0x77] * 8, n), pad([0x77] * 8, n), 120, 120, 0x3F800000, 1))
    return c


def ml_case(nrng, L, fmt):
    """Realistic MX operand blocks: Gaussian x per-channel outlier gain, MX-quantised."""
    gain = np.exp(nrng.normal(0, 1.0, size=32))
    A = nrng.normal(0, 1.0, size=32) * gain
    B = nrng.normal(0, 1.0, size=32) * np.exp(nrng.normal(0, 0.5, size=32))
    qa, ea = mq.mx_quantize(A, fmt)
    qb, eb = mq.mx_quantize(B, fmt)
    a = [int(x) for x in mq.fp8_encode(qa, fmt)][:L]
    b = [int(x) for x in mq.fp8_encode(qb, fmt)][:L]
    xa, xb = int(ea[0] + 127), int(eb[0] + 127)
    acc = int(np.array([nrng.normal(0, 8)], dtype=np.float32).view(np.uint32)[0])
    return a, b, xa, xb, acc, fmt


def make(n, L, seed, nrng):
    rng = random.Random(seed)
    cases = directed(L)
    while len(cases) < n:
        r = rng.random()
        if r < 0.45:
            cases.append(rand_case(rng, L))
        else:
            cases.append(ml_case(nrng, L, rng.getrandbits(1)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unit", type=int, default=20000)
    ap.add_argument("--n-blk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "sim", "vectors"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nrng = np.random.default_rng(args.seed)

    for name, L, n, fn in (("unit_vectors.mem", 8, args.n_unit, unit_line),
                           ("blk_vectors.mem", 32, args.n_blk, blk_line)):
        cases = make(n, L, args.seed + L, nrng)
        bad = 0
        with open(os.path.join(args.out, name), "w") as f:
            for (a, b, xa, xb, acc, fmt) in cases:
                exp = mxdotp_ref(a, b, xa, xb, acc, fmt)
                if L == 8 and mxdotp_hw(a, b, xa, xb, acc, fmt) != exp:
                    bad += 1
                if L == 32:
                    from rtl_model import dot8_hw, addround_hw
                    T, S, n1, p1, ng1, z1 = dot8_hw(a, b, xa, xb, fmt)
                    if addround_hw(T, S, acc, n1, p1, ng1, z1) != exp:
                        bad += 1
                f.write(fn(a, b, xa, xb, acc, fmt, exp) + "\n")
        print(f"{name}: {len(cases)} vectors written ({bad} model/reference mismatches)")
        assert bad == 0


if __name__ == "__main__":
    main()
