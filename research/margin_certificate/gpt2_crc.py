"""Direct risk control on the margin: pick threshold lam on a score so that  E[ 1{flip AND score > lam} ] <= alpha
(conformal risk control, Angelopoulos et al.; loss is monotone in lam, finite-sample valid under exchangeability).
Compares which SCORE gives the most certified tokens at the same guarantee. Same data/splits style as gpt2_study.py."""
import numpy as np
from safetensors.numpy import load_file
import os; D = os.environ.get("GPT2_DIR", "gpt2")  # dir with model.safetensors + tokenizer.json (openai-community/gpt2)
W = load_file(f"{D}/model.safetensors"); W = np.asarray(W.get("wte.weight", W.get("transformer.wte.weight")), np.float32)
F = np.load("features.npz"); hA, cand, zA, argB, argR, seq = (F[k] for k in ["hA", "cand", "zA", "argB", "argR", "seq"])
N, K = cand.shape; a = cand[:, 0]; flipB, flipR = argB != a, argR != a
gap = zA[:, :1] - zA[:, 1:]; S = np.empty_like(gap); CS = np.empty_like(gap)
for i in range(N):
    u = (W[a[i]] - W[cand[i, 1:]]).astype(np.float64); h = hA[i].astype(np.float64)
    S[i] = np.abs(u * h).sum(1); CS[i] = np.linalg.norm(u, axis=1) * np.linalg.norm(h)
scores = {"raw margin": gap.min(1), "gap/S": (gap / S).min(1), "gap/CS": (gap / CS).min(1),
          "margin + top-2 only": gap[:, 0]}
def crc_lambda(score, flip, alpha):
    n = len(score); order = np.argsort(-score)                 # candidates for lam = each observed score, descending
    lam_grid = np.sort(score)
    # risk(lam) = mean(flip & score > lam); smallest lam with (n*R + 1)/(n+1) <= alpha
    fl = flip.astype(float); cum = np.cumsum(fl[np.argsort(score)][::-1])[::-1]  # flips with score >= sorted_score[i]
    risk_gt = np.append(cum[1:], 0.0) / n                      # flips with score > sorted_score[i]
    ok = (n * risk_gt + 1) / (n + 1) <= alpha
    return lam_grid[np.argmax(ok)] if ok.any() else np.inf
rng = np.random.default_rng(1); seqs = np.unique(seq)
for alpha in (0.01, 0.001):
    print(f"\n=== alpha={alpha}: certify token if score > lam,  lam from conformal risk control  (30 sequence-level splits) ===")
    print("score                 certified (mean±sd)   P(cert & flip) vs fp16 (max)    vs fp32 [transfer]   P(flip | cert)")
    for k, s in scores.items():
        c_, v_, r_, f_ = [], [], [], []
        for rep in range(30):
            cal = np.isin(seq, rng.choice(seqs, len(seqs) // 2, replace=False)); te = ~cal
            lam = crc_lambda(s[cal], flipB[cal], alpha); c = s > lam
            c_.append(c[te].mean()); v_.append((c & flipB)[te].mean()); r_.append((c & flipR)[te].mean()); f_.append(flipB[te][c[te]].mean() if c[te].any() else 0)
        print(f"{k:20s} {np.mean(c_):6.2%} ± {np.std(c_):.2%}     {np.mean(v_):.4%} ({np.max(v_):.4%})        {np.mean(r_):.4%} ({np.max(r_):.4%})     {np.mean(f_):.4%}")

print("\nthreshold lam on the raw logit margin, calibrated on ALL 96 sequences:")
for alpha in (0.01, 0.001):
    lam = crc_lambda(scores["raw margin"], flipB, alpha)
    print(f"  alpha={alpha}: certify if margin > {lam:.3f} nats  -> {np.mean(scores['raw margin'] > lam):.2%} of tokens")
print(f"median margin {np.median(scores['raw margin']):.2f} nats; flip rate among tokens with margin < 0.5: {flipB[scores['raw margin'] < .5].mean():.1%}")
