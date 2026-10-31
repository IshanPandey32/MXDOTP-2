"""
mx_ref.py - Exact (arbitrary precision) golden reference for MXFP8 dot products.

Computes   result = RNE_fp32( acc + 2^(Xa+Xb-254) * sum_i A_i*B_i )   with ONE rounding,
using Python Fractions, so it is exact by construction.  Special values (NaN/Inf, E8M0 NaN
scale = 0xFF) follow the same policy as the RTL (canonical qNaN = 0x7FC00000).

Formats:  fmt = 0 -> E5M2 (bias 15, IEEE-like Inf/NaN)
          fmt = 1 -> E4M3 (bias 7, OCP variant: no Inf, NaN = S.1111.111)
"""
from fractions import Fraction

QNAN = 0x7FC00000
PINF = 0x7F800000
NINF = 0xFF800000


def fp8_decode(x, fmt):
    """-> (kind, sign, value) ; kind in {'fin','inf','nan'}"""
    s = (x >> 7) & 1
    if fmt == 0:
        e, m = (x >> 2) & 31, x & 3
        if e == 31:
            return ('nan' if m else 'inf'), s, None
        v = Fraction(m, 4) * Fraction(2) ** -14 if e == 0 else Fraction(4 + m, 4) * Fraction(2) ** (e - 15)
    else:
        e, m = (x >> 3) & 15, x & 7
        if e == 15 and m == 7:
            return 'nan', s, None
        v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else Fraction(8 + m, 8) * Fraction(2) ** (e - 7)
    return 'fin', s, (-v if s else v)


def fp32_decode(b):
    s, e, m = (b >> 31) & 1, (b >> 23) & 255, b & 0x7FFFFF
    if e == 255:
        return ('nan' if m else 'inf'), s, None
    v = Fraction(m) * Fraction(2) ** -149 if e == 0 else Fraction(8388608 + m) * Fraction(2) ** (e - 150)
    return 'fin', s, (-v if s else v)


def round_fp32(x, zero_sign=0):
    """Fraction -> FP32 bit pattern, round-to-nearest-even, gradual underflow, overflow -> Inf."""
    if x == 0:
        return zero_sign << 31
    s = 1 if x < 0 else 0
    mag = -x if s else x
    e = mag.numerator.bit_length() - mag.denominator.bit_length()   # e = floor(log2(mag))
    if Fraction(2) ** e > mag:
        e -= 1
    elif Fraction(2) ** (e + 1) <= mag:
        e += 1
    L = max(e - 23, -149)                      # exponent of the result LSB
    scaled = mag / Fraction(2) ** L
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if n >= (1 << 24):
        n >>= 1
        L += 1
    if n >= (1 << 23):
        ebits = L + 23 + 127
        mant = n - (1 << 23)
    else:
        ebits, mant = 0, n
    if ebits >= 255:
        return (s << 31) | PINF
    return (s << 31) | (ebits << 23) | mant


def mxdotp_ref(a, b, xa, xb, acc, fmt):
    """a, b : lists of FP8 bytes (any length: 8 for one MXDOTP, 32 for a full MX block)."""
    if xa == 0xFF or xb == 0xFF:
        return QNAN
    nan = False
    pinf = ninf = False
    all_neg_zero = True
    total = Fraction(0)
    for x, y in zip(a, b):
        kx, sx, vx = fp8_decode(x, fmt)
        ky, sy, vy = fp8_decode(y, fmt)
        neg = sx ^ sy
        if kx == 'nan' or ky == 'nan':
            nan = True
            all_neg_zero = False
            continue
        zx = kx == 'fin' and vx == 0
        zy = ky == 'fin' and vy == 0
        if (kx == 'inf' and zy) or (ky == 'inf' and zx):
            nan = True
            all_neg_zero = False
            continue
        if kx == 'inf' or ky == 'inf':
            if neg:
                ninf = True
            else:
                pinf = True
            all_neg_zero = False
            continue
        total += vx * vy
        if not ((zx or zy) and neg):
            all_neg_zero = False
    kacc, sacc, vacc = fp32_decode(acc)
    if kacc == 'inf':
        if sacc:
            ninf = True
        else:
            pinf = True
    if kacc == 'nan' or nan or (pinf and ninf):
        return QNAN
    if pinf:
        return PINF
    if ninf:
        return NINF
    S = xa + xb - 254
    exact = vacc + total * Fraction(2) ** S
    zero_sign = 1 if (vacc == 0 and sacc and all_neg_zero) else 0
    return round_fp32(exact, zero_sign)
