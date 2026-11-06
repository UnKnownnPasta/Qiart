"""Real-GPT-2 study of margin certificates: bf16 (impl A) vs fp16 (impl B), teacher-forced on real prose.
 1. Calibrated certificates: absolute error budget  vs  Cauchy-Schwarz scale  vs  cancellation scale S=sum|u_i h_i|.
    Repeated random splits BY SEQUENCE give mean +- sd of certified fraction and validity of the guarantee.
 2. Does the cancellation-normalised margin predict flips better than the raw margin? (AUC, Brier after calibration)
 3. Transfer: certificate calibrated on A-vs-fp16 applied to flips of A-vs-fp32.
"""
import numpy as np
from safetensors.numpy import load_file
from certify import gamma, U
import os; D = os.environ.get("GPT2_DIR", "gpt2")  # dir with model.safetensors + tokenizer.json (openai-community/gpt2)
W = load_file(f"{D}/model.safetensors"); W = np.asarray(W.get("wte.weight", W.get("transformer.wte.weight")), np.float32)
F = np.load("features.npz"); hA, hB, hR, cand, zA, argB, argR, seq = (F[k] for k in ["hA", "hB", "hR", "cand", "zA", "argB", "argR", "seq"])
N, K = cand.shape; d = hA.shape[1]; a = cand[:, 0]
flipB, flipR = argB != a, argR != a
print(f"positions {N} ({len(np.unique(seq))} sequences)  flip rate A-vs-fp16 {flipB.mean():.3%}  A-vs-fp32 {flipR.mean():.3%}")
print(f"flips landing outside A's top-{K}: {int((~(argB[:, None] == cand).any(1) & flipB).sum())}")

gap = zA[:, :1] - zA[:, 1:]
S = np.empty_like(gap); CS = np.empty_like(gap); dB = np.empty_like(gap); dR = np.empty_like(gap)
for i in range(N):
    u = (W[a[i]] - W[cand[i, 1:]]).astype(np.float64); h = hA[i].astype(np.float64)
    S[i] = np.abs(u * h).sum(1); CS[i] = np.linalg.norm(u, axis=1) * np.linalg.norm(h)
    dB[i] = u @ (hB[i] - h); dR[i] = u @ (hR[i] - h)
head = 2 * gamma(d - 1, U["fp32"])                     # rigorous fp32-accumulation term for the lm_head, times S

def conf_q(v, alpha):
    n = len(v); return np.sort(v)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1]
tB = {"abs budget": np.abs(dB).max(1), "Cauchy-Schwarz": (np.abs(dB) / CS).max(1), "cancellation S": (np.abs(dB) / S).max(1)}
def certify(name, q): return {"abs budget": gap > q, "Cauchy-Schwarz": gap > q * CS, "cancellation S": gap > (q + head) * S}[name].all(1)

rng = np.random.default_rng(0); seqs = np.unique(seq)
for alpha in (0.01, 0.001):
    out = {k: [] for k in tB}; viol = {k: [] for k in tB}; violR = {k: [] for k in tB}; diff = []
    for rep in range(30):
        cal_s = rng.choice(seqs, len(seqs) // 2, replace=False); cal = np.isin(seq, cal_s); te = ~cal
        for k in tB:
            c = certify(k, conf_q(tB[k][cal], alpha))
            out[k].append(c[te].mean()); viol[k].append((c & flipB)[te].sum() / te.sum()); violR[k].append((c & flipR)[te].sum() / te.sum())
        diff.append(out["cancellation S"][-1] - out["abs budget"][-1])
    print(f"\n=== alpha={alpha}: 30 random sequence-level splits (calibrate on half, test on half) ===")
    print("method               certified (mean±sd)   P(cert & flip) vs fp16   P(cert & flip) vs fp32 [transfer]")
    for k in tB: print(f"{k:20s} {np.mean(out[k]):6.2%} ± {np.std(out[k]):.2%}     {np.mean(viol[k]):.4%} (max {np.max(viol[k]):.4%})   {np.mean(violR[k]):.4%} (max {np.max(violR[k]):.4%})")
    print(f"S minus abs-budget certified fraction: {np.mean(diff)*100:+.2f} pts  (sd {np.std(diff)*100:.2f}, min {np.min(diff)*100:+.2f}, max {np.max(diff)*100:+.2f})")

# ---- 2. predicting flips: raw margin vs cancellation-normalised margin
def auc(score, y):                                    # P(score_pos > score_neg), ties 0.5
    r = np.argsort(np.argsort(score)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
feats = {"raw margin  gap_min": np.log(gap[:, 0] + 1e-9) * 0 + np.log(gap.min(1) + 1e-9),
         "gap/S  (min over j)": np.log((gap / S).min(1) + 1e-12),
         "gap/CS (min over j)": np.log((gap / CS).min(1) + 1e-12)}
print("\n=== flip prediction (A vs fp16), higher AUC = better ranking of fragile tokens ===")
cal = np.isin(seq, rng.choice(seqs, len(seqs) // 2, replace=False)); te = ~cal
for k, f in feats.items():
    # calibrated probability = binned empirical flip rate on calibration half (20 equal-count bins), Brier on test half
    edges = np.quantile(f[cal], np.linspace(0, 1, 21)); b_c = np.clip(np.searchsorted(edges, f[cal]) - 1, 0, 19); b_t = np.clip(np.searchsorted(edges, f[te]) - 1, 0, 19)
    rate = np.array([flipB[cal][b_c == i].mean() for i in range(20)]); p = rate[b_t]
    brier = np.mean((p - flipB[te]) ** 2); base = np.mean((flipB[cal].mean() - flipB[te]) ** 2)
    ll = -np.mean(flipB[te] * np.log(np.clip(p, 1e-4, 1)) + (~flipB[te]) * np.log(np.clip(1 - p, 1e-4, 1)))
    print(f"{k:22s} AUC(test) {auc(-f[te], flipB[te]):.4f}   Brier skill {1 - brier / base:.3f}   log-loss {ll:.4f}")
