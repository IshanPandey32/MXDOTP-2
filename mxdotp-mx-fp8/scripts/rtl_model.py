"""
rtl_model.py - Bit-accurate Python MIRROR of the Verilog datapath (same widths, same
algorithm: fixed-point window + sticky + single RNE).  It proves the *algorithm* against the
exact reference in mx_ref.py independently of RTL simulation.

Run:  python3 scripts/rtl_model.py      (self-check with 200k random vectors)
"""
import random
from mx_ref import mxdotp_ref, QNAN, PINF, NINF

OT, DMAX, SW, TW = 32, 73, 132, 72      # must match localparams in rtl/common/mxdotp_addround.v


def dec(x, fmt):
    s = (x >> 7) & 1
    if fmt == 0:
        e, m = (x >> 2) & 31, x & 3
        nan, inf = (e == 31 and m != 0), (e == 31 and m == 0)
        M = ((1 if e else 0) << 3) | (m << 1)
        eb = 1 if e == 0 else e
    else:
        e, m = (x >> 3) & 15, x & 7
        nan, inf = (e == 15 and m == 7), False
        M = ((1 if e else 0) << 3) | m
        eb = (1 if e == 0 else e) + 8
    zero = (M == 0) and not nan and not inf
    return s, M, eb, zero, nan, inf


def dot8_hw(a, b, xa, xb, fmt):
    """Stage 1: decode + 4x4 multipliers + shift + signed sum -> (T, S, flags)."""
    T = 0
    nan_f = (xa == 255) or (xb == 255)
    pinf = ninf = False
    allnegz = True
    for x, y in zip(a, b):
        sa, ma, ea, za, na, ia = dec(x, fmt)
        sb, mb, eb, zb, nb, ib = dec(y, fmt)
        fin = not (na or nb or ia or ib)
        lnan = na or nb or (ia and zb) or (ib and za)
        linf = (ia or ib) and not lnan
        neg = sa ^ sb
        pos = ea + eb - 2
        pm = ((ma * mb) << pos) if fin else 0
        T += -pm if neg else pm
        lzero = fin and (za or zb)
        nan_f |= lnan
        pinf |= linf and not neg
        ninf |= linf and neg
        allnegz &= bool(lzero and neg)
    S = xa + xb - 254
    assert -(1 << (TW - 1)) <= T < (1 << (TW - 1))
    return T, S, nan_f, pinf, ninf, allnegz


def addround_hw(T, S, acc, nan_f, pinf_f, ninf_f, allnegz):
    sa, ea, ma = acc >> 31, (acc >> 23) & 255, acc & 0x7FFFFF
    acc_nan, acc_inf = (ea == 255 and ma != 0), (ea == 255 and ma == 0)
    inf_pos = pinf_f or (acc_inf and not sa)
    inf_neg = ninf_f or (acc_inf and sa)
    if nan_f or acc_nan or (inf_pos and inf_neg):
        return QNAN
    if inf_pos:
        return PINF
    if inf_neg:
        return NINF
    # ---------------- stage 2 : align + add ----------------
    ea_eff = 1 if ea == 0 else ea
    A = ((1 if ea else 0) << 23) | ma
    if T == 0:                                   # bypass: acc + 0 == acc
        if (acc & 0x7FFFFFFF) == 0:
            return ((sa & int(allnegz)) << 31)
        return acc
    d = ea_eff - S - 116
    Tw = T << OT
    if A == 0:
        Au, wexp = 0, S - 34 - OT
    elif d > DMAX:                               # acc dominates: product sum -> +/-1 sticky
        Au = A << (OT + DMAX)
        Tw = -1 if T < 0 else 1
        wexp = ea_eff - 150 - OT - DMAX
    else:
        wexp = S - 34 - OT
        pa = OT + d
        if pa >= 0:
            Au = A << pa
        else:
            k = -pa
            if k >= 24:
                Au = 1
            else:
                Au = A >> k
                if A & ((1 << k) - 1):
                    Au |= 1
    W = (-Au if sa else Au) + Tw
    assert -(1 << (SW - 1)) <= W < (1 << (SW - 1))
    sign = 1 if W < 0 else 0
    mag = -W if sign else W
    # ---------------- stage 3 : normalise + round ----------------
    if mag == 0:
        return 0
    p = mag.bit_length() - 1
    eb = p + wexp + 127
    r = max(p - 23, -149 - wexp)
    sh = min(r + 26, SW + 28)
    ext = mag << 28
    K = ext >> sh
    assert K >> 26 == 0
    lost = (ext & ((1 << sh) - 1)) != 0
    keep, guard = (K >> 2) & 0xFFFFFF, (K >> 1) & 1
    sticky = bool(K & 1) or lost
    inc = 1 if (guard and (sticky or (keep & 1))) else 0
    rounded = keep + inc
    base = eb - 1 if eb >= 1 else 0
    packed = (base << 23) + rounded
    if (packed >> 23) >= 255:
        return (sign << 31) | PINF
    return (sign << 31) | (packed & 0x7FFFFFFF)


def mxdotp_hw(a, b, xa, xb, acc, fmt):
    T, S, n, p, ng, z = dot8_hw(a, b, xa, xb, fmt)
    return addround_hw(T, S, acc, n, p, ng, z)


# ------------------------------------------------------------------ self check
def rand_fp8(rng, fmt, kind):
    if kind == 0:
        return rng.getrandbits(8)
    e = rng.randint(0, 31 if fmt == 0 else 15)
    return (rng.getrandbits(1) << 7) | ((e << 2 | rng.getrandbits(2)) if fmt == 0 else (e << 3 | rng.getrandbits(3)))


def rand_acc(rng):
    k = rng.randint(0, 5)
    if k == 0:
        return rng.getrandbits(32)
    if k == 1:
        return rng.getrandbits(1) << 31 | rng.getrandbits(23)          # subnormal / zero
    if k == 2:
        return 0
    e = rng.randint(90, 170)
    return rng.getrandbits(1) << 31 | e << 23 | rng.getrandbits(23)


def rand_case(rng, L=None):
    fmt = rng.getrandbits(1)
    L = L or (8 if rng.random() < 0.6 else 32)
    kind = rng.randint(0, 3)
    a = [rand_fp8(rng, fmt, kind & 1) for _ in range(L)]
    b = [rand_fp8(rng, fmt, kind & 1) for _ in range(L)]
    if rng.random() < 0.15:                                           # force cancellation
        a = [a[j] if j % 2 == 0 else a[j - 1] for j in range(L)]
        b = [b[j] if j % 2 == 0 else b[j - 1] ^ 0x80 for j in range(L)]
    sc = rng.random()
    if sc < 0.6:
        xa, xb = rng.randint(110, 140), rng.randint(110, 140)
    elif sc < 0.9:
        xa, xb = rng.randint(0, 254), rng.randint(0, 254)
    else:
        xa, xb = rng.choice([0, 1, 254, 255, 127]), rng.choice([0, 1, 254, 255, 127])
    return a, b, xa, xb, rand_acc(rng), fmt


def selfcheck(n=200000, seed=1):
    rng = random.Random(seed)
    bad = 0
    for _ in range(n):
        a, b, xa, xb, acc, fmt = rand_case(rng)
        ref = mxdotp_ref(a, b, xa, xb, acc, fmt)
        got = mxdotp_hw(a, b, xa, xb, acc, fmt)
        if ref != got:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH fmt={fmt} L={len(a)} xa={xa} xb={xb} acc={acc:08x} ref={ref:08x} hw={got:08x}")
    print(f"rtl_model self-check: {n} vectors, {bad} mismatches")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if selfcheck() else 0)
