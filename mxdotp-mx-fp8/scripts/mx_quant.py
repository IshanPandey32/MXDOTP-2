"""
mx_quant.py - Small NumPy emulation of OCP Microscaling (MX) quantisation (MXFP8).

  * quant_fp8      : round-to-nearest-even into E5M2 / E4M3 (saturating)
  * mx_quantize    : block (default 32) shared power-of-two scale (E8M0) + FP8 elements
  * mx_dequantize  : elements * 2^scale_exp
  * fp8_encode     : FP8 values -> hardware byte codes (same encoding as the RTL)
Scale rule (OCP MX spec):  shared_exp = floor(log2(max|x| in block)) - emax_elem
                           E4M3 emax = 8 (max 448),  E5M2 emax = 15 (max 57344)
"""
import numpy as np
from mx_ref import fp8_decode

SPEC = {
    0: dict(name="E5M2", mbits=2, max=57344.0, min_exp=-14, emax=15),
    1: dict(name="E4M3", mbits=3, max=448.0, min_exp=-6, emax=8),
}


def quant_fp8(x, fmt):
    sp = SPEC[fmt]
    x = np.asarray(x, dtype=np.float64)
    neg = np.signbit(x)
    ax = np.minimum(np.abs(x), sp["max"])
    e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))
    e = np.maximum(e, sp["min_exp"])
    step = np.exp2(e - sp["mbits"])
    q = np.minimum(np.rint(ax / step) * step, sp["max"])      # rint = ties-to-even
    return np.where(neg, -q, q)


def mx_quantize(x, fmt, block=32):
    """x: (..., K) with K % block == 0.  Returns (elements, scale_exponents)."""
    x = np.asarray(x, dtype=np.float64)
    shp = x.shape
    nb = shp[-1] // block
    xb = x.reshape(shp[:-1] + (nb, block))
    amax = np.max(np.abs(xb), axis=-1, keepdims=True)
    sexp = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - SPEC[fmt]["emax"]
    sexp = np.clip(sexp, -127, 127)
    q = quant_fp8(xb / np.exp2(sexp), fmt)
    return q.reshape(shp), sexp[..., 0].astype(np.int64)


def mx_dequantize(q, sexp, block=32):
    shp = q.shape
    qb = q.reshape(shp[:-1] + (shp[-1] // block, block))
    return (qb * np.exp2(sexp[..., None])).reshape(shp)


def per_tensor_fp8(x, fmt):
    """Baseline: ONE scale for the whole tensor (what MX improves upon)."""
    amax = np.max(np.abs(x))
    sexp = np.floor(np.log2(amax)) - SPEC[fmt]["emax"]
    return quant_fp8(x / 2.0 ** sexp, fmt) * 2.0 ** sexp


_tables = {}


def _table(fmt):
    if fmt not in _tables:
        mags, codes = [], []
        for c in range(128):
            kind, _, v = fp8_decode(c, fmt)
            if kind == "fin":
                mags.append(float(v))
                codes.append(c)
        order = np.argsort(mags, kind="stable")
        _tables[fmt] = (np.array(mags)[order], np.array(codes)[order])
    return _tables[fmt]


def fp8_encode(q, fmt):
    """Quantised FP8 values -> uint8 codes (exact match required)."""
    mags, codes = _table(fmt)
    q = np.asarray(q, dtype=np.float64)
    aq = np.abs(q)
    idx = np.searchsorted(mags, aq)
    idx = np.clip(idx, 0, len(mags) - 1)
    assert np.all(mags[idx] == aq), "value is not an exact FP8 number"
    return (codes[idx] | (np.signbit(q).astype(np.int64) << 7)).astype(np.uint8)
