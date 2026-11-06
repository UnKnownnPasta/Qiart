# Margin certificates for LLM token flips (prototype)

`certify.py` has the bound and the flip-probability model (derivation in its docstring).
`experiment.py` checks them on synthetic data; `results.txt` is its output.

Run (needs `numpy` and `ml_dtypes`):

    python experiment.py 1.0

What it tests: implementation A rounds the hidden state to bf16 and accumulates in fp32 in natural order;
implementation B rounds to fp16 and accumulates in fp32 in a random order. A token is "certified" when its
observed gap to every runner-up exceeds the deterministic radius.

Findings (20,000 synthetic positions per setting, d=2048, vocab 4000):

| setting  | certified | flips among certified | actual flip rate | model-predicted |
|----------|-----------|-----------------------|------------------|-----------------|
| gaussian | 49.7%     | 0                     | 0.325%           | 0.476%          |
| outlier  | 65.7%     | 0                     | 0.330%           | 0.401%          |

Caveats: synthetic random weights and hidden states, not a real model; head-only (body error not bounded);
first-order bound; the probability model over-predicts, so it is safe but not tight.

Next step if pursued: run the same check on real logits from a small open model, and add a bound on
perturbation of h propagated from the transformer body.
