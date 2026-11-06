"""
Synthetic validation of the margin certificate.

Implementation A: hidden state rounded to bf16, fp32 accumulation, natural order.
Implementation B: hidden state rounded to fp16, fp32 accumulation, random order.
Checks (1) certified tokens NEVER flip, (2) how much of the sequence is certified,
(3) calibration of the flip-probability model on uncertified tokens.
"""
import sys, numpy as np, ml_dtypes
from certify import *

def acc32(prod, order=None):
    """Sequential float32 accumulation of exact products along last axis, optionally permuted."""
    if order is not None:
        prod = prod[..., order]
    return np.cumsum(prod.astype(np.float32), axis=-1, dtype=np.float32)[..., -1]

def run(kind, N=20000, d=2048, V=4000, K=4, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((V, d)) / np.sqrt(d) * scale).astype(np.float32)
    W = W.astype(ml_dtypes.bfloat16).astype(np.float32)           # shared bf16 weights (exact in both)
    outl = np.ones(d); 
    if kind == "outlier":
        outl[rng.choice(d, 8, replace=False)] = 25.0               # "massive activation" channels
    perm = rng.permutation(d)
    flips = cert_viol = n_cert = 0
    rows = []
    for s in range(0, N, 2000):
        h = (rng.standard_normal((2000, d)) * outl).astype(np.float64)
        lg = h.astype(np.float32) @ W.T
        top = np.argpartition(-lg, K, axis=1)[:, :K]
        hA = h.astype(ml_dtypes.bfloat16).astype(np.float32)
        hB = h.astype(np.float16).astype(np.float32)
        for i in range(2000):
            idx = top[i]
            Wk = W[idx]                                            # K x d
            gA = acc32(Wk * hA[i]); gB = acc32(Wk * hB[i], perm)   # K logits each
            # reference order by A's observed logits
            order = np.argsort(-gA); a = order[0]
            aB = np.argmax(gB)
            flip = aB != a
            # certificate: for every runner-up j, gap_A(a,j) > r(a,j)
            ok = True; pnf = 1.0; 
            for j in order[1:]:
                u = (Wk[a] - Wk[j]).astype(np.float64)
                S, ss = fragility(u, hA[i].astype(np.float64))
                r = certified_radius(S, d, "bf16", "fp16")
                sg = noise_sigma(ss, d, "bf16", "fp16")
                gap = float(gA[a] - gA[j])
                if gap <= r: ok = False
                pnf *= 1.0 - float(flip_probability(gap, sg))
            rows.append((ok, flip, 1.0 - pnf))
    ok, fl, p = map(np.array, zip(*rows))
    print(f"\n=== {kind}: N={len(ok)}  actual flip rate = {fl.mean():.4%}")
    print(f"certified tokens        : {ok.mean():.2%}")
    print(f"flips among certified   : {fl[ok].sum()}  (must be 0)")
    print(f"flips among uncertified : {fl[~ok].sum()} of {(~ok).sum()}  = {fl[~ok].mean():.3%}")
    print(f"model-predicted flip rate overall = {p.mean():.4%}  vs actual {fl.mean():.4%}")
    bins = [0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0001]
    print("predicted-p bin        n    mean_pred  empirical")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): print(f"[{lo:.4f},{hi:.4f}) {m.sum():7d}  {p[m].mean():9.5f}  {fl[m].mean():9.5f}")
    return ok, fl, p

if __name__ == "__main__":
    for kind in ("gaussian", "outlier"):
        run(kind, scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
