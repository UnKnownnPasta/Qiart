# Margin certificates for LLM token flips

**Question:** when do two inference stacks (here bf16 vs fp16) pick the same greedy token, and can we say
which tokens are safe *in advance*, with a guarantee?

## What is known (so this is not claimed as new)
A margin lemma already exists: if logit error is below half the margin, the argmax is unchanged (e.g. a
2025 ResearchGate preprint on batch invariance and seed locking, which I could only read in search fragments —
verify its exact statement). Empirical papers show flips concentrate at small top-2 margins
([cross-precision](https://arxiv.org/pdf/2609.26621), [cross-GPU](https://arxiv.org/abs/2609.25624)).
[Lipschitz certification under floating point](https://arxiv.org/abs/2603.13334) covers feed-forward ReLU
classifiers, not transformers.

## What I tested
1. `certify.py`, `experiment.py`, `results.txt` — first-order head-only rounding bound, synthetic data.
   Held up (0 violations) but **synthetic only**; superseded by the real-model results below.
2. Real GPT-2 small (124M), 24,576 teacher-forced positions of technical prose, bf16 vs fp16 bodies with
   fp32 LayerNorm/softmax/GELU internals and fp32-accumulated lm_head. Sanity: fp32 / bf16 / fp16 NLL =
   2.971 / 2.950 / 2.973.
   - `gpt2_features.py` caches hidden states; `gpt2_study.py` and `gpt2_crc.py` analyse them.

## Findings on real GPT-2 (all positions, one corpus, one model, one implementation pair)
- **Flip rate is large:** greedy token differs bf16-vs-fp16 at **3.97%** of positions (bf16-vs-fp32: 3.94%, so
  bf16 is the source; fp16 ≈ fp32). In the synthetic setup it was 0.3%.
- **Error-budget certificates are valid but conservative.** Calibrated by split-conformal on held-out
  sequences, the guarantee holds. At alpha=1% only 77% of tokens certify (abs. budget), 66% (Cauchy-Schwarz),
  79% (my cancellation scale S=Σ|uᵢhᵢ|). The S gain over the plain budget is **+2.5 points (sd 0.4)** at
  alpha=1% and not detectable at alpha=0.1% (+1.0, sd 2.6). Realised violation rate is ~0.02% vs 1% allowed.
- **The per-token cancellation idea mostly does not pay off.** The raw margin predicts flips as well as the
  S-normalised one (AUC 0.959 vs 0.959; Cauchy-Schwarz-normalised 0.962). Body (transformer-layer) error
  dominates the head rounding error.
- **My Gaussian flip-probability model failed.** With noise scale calibrated on held-out data it predicted
  1.5% flips vs 3.9% actual, and low-probability bins had empirical flip rates 10–100x higher (heavy-tailed,
  systematic bf16 body error). The synthetic "conservative" result did not transfer. Treat any parametric
  flip model as wrong until calibrated nonparametrically.
- **What works: calibrate the margin threshold directly (conformal risk control, `gpt2_crc.py`).**
  Certify a token if its top-2 margin exceeds λ. On all 96 sequences: alpha=1% → λ=0.158 nats, 91.2% of tokens
  certified; alpha=0.1% → λ=0.366 nats, 81.2% certified. Over 30 sequence-level splits, P(certified & flip) on
  held-out sequences averaged 0.96% (alpha=1%) and 0.098% (alpha=0.1%) — tight, but the guarantee is in
  expectation, so individual splits reach up to 1.38% and 0.23%. The threshold calibrated on bf16-vs-fp16
  also controlled risk for flips vs fp32 (0.92%, 0.103%). Tokens with margin < 0.5 nats flip 16% of the time.
  The choice of score (raw margin, gap/S, gap/CS) makes no practical difference.

## Caveats
- One small model, one corpus, teacher-forced (not free-running generation). Guarantee assumes calibration and
  deployment positions are exchangeable; positions inside one sequence are not independent, and domain shift
  would break it. Only the top-50 candidates are checked (0 flips landed outside them here).
- It certifies one-position-ahead agreement given identical context. A response diverges if *any* token flips
  (then context differs), and the sequence-level divergence prediction is **untested**.

## Next steps
Free-running generation: predict P(two stacks produce the same N-token completion) from per-token calibrated
risks and check it against actual divergence; test a larger model and a second hardware/kernel pair; test
domain shift (calibrate on prose, deploy on code).

Run: `GPT2_DIR=<dir with model.safetensors, tokenizer.json> python gpt2_features.py 96`, then
`python gpt2_study.py`, `python gpt2_crc.py` (needs jax, numpy, safetensors, tokenizers).
