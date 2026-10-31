"""
accuracy_study.py - Numerical study (the "ML side" of the project).

Question: for realistic ML-like tensors (Gaussian with outlier channels), how much error do
  (1) MX quantisation itself (E4M3 / E5M2, block 32)  vs  per-tensor FP8 scaling,
  (2) the accumulation scheme of the hardware           add on top?

Accumulation schemes compared on identical MX-quantised operands:
  SW   FP8->FP32 software baseline: sequential FP32 FMA per 32-block, scale applied after
  B8   MXDOTP (paper)           : exact sum of 8 products + acc, ONE FP32 rounding per 8 elements
  B32  MXDOTP-B32 (improved)    : exact sum of 32 products + acc, ONE FP32 rounding per 32 elements
Reference ("ideal") = float64 accumulation of the same quantised operands.

NOTE: B8/B32 are emulated with float64 chunk sums (exact for practically all chunks) followed by
one float32 rounding, which mirrors the RTL's single-rounding behaviour to within double-rounding
effects (<<1 ulp).  The RTL itself is verified bit-exactly by the testbenches.

Usage: python3 scripts/accuracy_study.py [--M 16 --N 16 --K 4096 --seed 0 --sigma 1.0]
"""
import argparse
import numpy as np
import mx_quant as mq

f32 = np.float32


def make_data(M, K, N, rng, sigma=1.0):
    A = rng.normal(size=(M, K)) * np.exp(rng.normal(0, sigma, size=K))        # outlier channels (sigma = strength)
    B = rng.normal(size=(K, N)) * np.exp(rng.normal(0, 0.5, size=(K, 1)))
    return A, B


def rel_rms(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def study(fmt, M, K, N, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    A, B = make_data(M, K, N, rng, sigma)
    exact = A @ B

    # ---- quantisation error: per-tensor FP8 vs MX (both accumulated in float64) ----
    pt = mq.per_tensor_fp8(A, fmt) @ mq.per_tensor_fp8(B, fmt)
    qa, ea = mq.mx_quantize(A, fmt)
    qbT, eb = mq.mx_quantize(B.T, fmt)
    nb = K // 32
    qa_b = qa.reshape(M, nb, 32)
    qb_b = qbT.reshape(N, nb, 32)
    scale = np.exp2(ea[:, :, None] + eb.T[None, :, :])                        # (M, nb, N)
    blocks = np.einsum("mbk,nbk->mbn", qa_b, qb_b) * scale                    # exact block sums
    ideal = blocks.sum(axis=1)

    # ---- accumulation schemes ----
    # SW : FP8->FP32, sequential FP32 FMA inside each block, scale after, FP32 accumulate
    acc_sw = np.zeros((M, N), dtype=f32)
    for b in range(nb):
        part = np.zeros((M, N), dtype=f32)
        for k in range(32):
            prod = qa_b[:, b, k][:, None] * qb_b[:, b, k][None, :]            # exact in float64
            part = (part.astype(np.float64) + prod).astype(f32)               # fused: 1 rounding
        acc_sw = (acc_sw.astype(np.float64) + part.astype(np.float64) * scale[:, b, :]).astype(f32)

    # B8 : one rounding per 8 elements
    chunks8 = np.einsum("mbck,nbck->mbcn", qa_b.reshape(M, nb, 4, 8), qb_b.reshape(N, nb, 4, 8)) \
        * scale[:, :, None, :]
    acc8 = np.zeros((M, N), dtype=f32)
    for b in range(nb):
        for c in range(4):
            acc8 = (acc8.astype(np.float64) + chunks8[:, b, c, :]).astype(f32)

    # B32 : one rounding per 32 elements
    acc32 = np.zeros((M, N), dtype=f32)
    for b in range(nb):
        acc32 = (acc32.astype(np.float64) + blocks[:, b, :]).astype(f32)

    return dict(
        fmt=mq.SPEC[fmt]["name"],
        q_pertensor=rel_rms(pt, exact),
        q_mx=rel_rms(ideal, exact),
        acc_sw=rel_rms(acc_sw.astype(np.float64), ideal),
        acc_b8=rel_rms(acc8.astype(np.float64), ideal),
        acc_b32=rel_rms(acc32.astype(np.float64), ideal),
        total_b32=rel_rms(acc32.astype(np.float64), exact),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.0, help="log-normal outlier-channel strength")
    args = ap.parse_args()
    assert args.K % 32 == 0
    print(f"C = A({args.M}x{args.K}) @ B({args.K}x{args.N}),  Gaussian x outlier channels (sigma {args.sigma}), seed {args.seed}\n")
    print("Quantisation error (relative RMS vs original FP64 matmul):")
    rows = [study(f, args.M, args.K, args.N, args.seed, args.sigma) for f in (1, 0)]
    for r in rows:
        print(f"  {r['fmt']}: per-tensor FP8 = {r['q_pertensor']:.3e}   MX (block 32) = {r['q_mx']:.3e}")
    print("\nAccumulation error added by the hardware scheme (relative RMS vs float64 of the SAME MX operands):")
    print(f"  {'fmt':5s} {'SW FP8->FP32':>14s} {'MXDOTP (B8)':>14s} {'MXDOTP-B32':>14s}")
    for r in rows:
        print(f"  {r['fmt']:5s} {r['acc_sw']:14.3e} {r['acc_b8']:14.3e} {r['acc_b32']:14.3e}")
    print("\nTotal error of the improved path vs original FP64 matmul:")
    for r in rows:
        print(f"  {r['fmt']}: {r['total_b32']:.3e}")


if __name__ == "__main__":
    main()
