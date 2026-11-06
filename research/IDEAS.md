# Research-gap ideas (2026-11-06)

Searched for gaps where a small piece of math plus software could close them. Each idea is pitched in the
"category gap" shape: *tools measure X, but what matters is Y.*

**How sure am I?** The gaps come from web-search snippets of 2026 preprints. I did not read the full papers, and
my novelty checks were a few searches each, not a literature review. Verify before claiming novelty.

---

## 1. "Safe token" certificates for LLM output  — prototype built

**One line:** The same prompt gives different answers on different GPUs. A token is safe if its lead over the
runner-up is bigger than the rounding noise times how much cancellation its logit gap involves.

**Category gap:** Existing work *measures* flip rates (empirically, or with a heuristic flip-probability model)
and *mitigates* them (deterministic kernels, margin-triggered re-verification). I found no per-token
certificate with a deterministic bound that a deployer can compute from the hidden state.

**Math:** gap = u·h, fragility S = Σ|uᵢhᵢ|. Any two stacks can only disagree on the sign of the gap if
|gap| ≤ (u_X + u_Y + γ_X + γ_Y)·S (Higham-style first-order bound, any summation order). Plus a CLT model for
P(flip) on uncertified tokens, and 1 − Π(1 − pₜ) for a whole response.

**Demo:** a response where each token is green (certified), amber, or red, with a "chance this answer differs on
other hardware" number.

**Result so far (synthetic only):** see `margin_certificate/results.txt`. 0 flips among certified tokens
(50–66% of tokens certified); the probability model is conservative (over-predicts flips by ~1.2–2.5x).

**Limits:** covers the output head only (body error enters as a perturbation on h and is not bounded yet);
synthetic weights because Hugging Face is blocked here; one pair of implementations tested.

Sources: [cross-GPU nondeterminism](https://arxiv.org/abs/2609.25624),
[greedy decoding not precision-invariant](https://arxiv.org/pdf/2609.26621),
[margin-triggered verification](https://arxiv.org/html/2605.30218v1),
[original NeurIPS study](https://arxiv.org/abs/2506.09501).

---

## 2. "Ownership score" for AI-assisted work  — closest in spirit to the UNCLTR deck

**One line:** Screen-time tools count how long you used AI. They don't tell you whether you still *own* what it
wrote for you.

**Category gap:** AI-coding tools report usage and speed. Anthropic's randomized trial found AI-assisted
learners scored ~50% vs ~65% on a later quiz with similar completion time, and reviews say the field lacks
instruments that pair behaviour logs with independent, AI-free measures of retention.

**Math (the part that makes it more than a survey):** treat each accepted suggestion as a "memory" with an
unknown half-life. Estimate per-concept retention with a Bayesian forgetting-curve model, and choose which
tiny AI-off probe to ask next (adaptive scheduling / bandit). Output: ownership = expected fraction of what
you accepted that you could reproduce after a delay.

**Build:** local-first (same privacy story as UNCLTR): browser or editor extension that logs accepted
suggestions on-device, plus 30-second recall probes.

**Not yet checked:** whether spaced-repetition-for-code tools already do this. Do that search first.

Sources: [Anthropic trial](https://www.anthropic.com/research/AI-assistance-coding-skills),
[Designing Against Deskilling](https://arxiv.org/pdf/2609.20143),
[offloading scoping review](https://pmc.ncbi.nlm.nih.gov/articles/PMC13405335/).

---

## 3. Per-step confidence doesn't compose: certified agent runs

**One line:** If each step is 90% reliable, a 15-step agent run is nowhere near 90% reliable.

**Category gap:** An EMNLP 2026 paper reports 83.5% miscoverage at a nominal 10% level on a 15-step agent when
conformal calibration is done per step. I found nothing giving guarantees that stay valid when the agent
stops at a data-dependent time.

**Math:** calibrate on the running maximum of a nonconformity score over whole trajectories (trajectories are
exchangeable even when steps are not), then optimize the per-step threshold shape for tightness. Validity at
any stopping time follows because the running max is nondecreasing.

**Caveat:** conformal bands are known technology; the contribution would be the stopping-time guarantee and
the LLM-agent application, so the math is less novel than idea 1.

Sources: [per-step calibration does not transfer](https://aclanthology.org/2026.emnlp-main.1808/),
[UQ for LLM agents](https://arxiv.org/html/2609.07395v1).

---

## Rejected

- **Leaderboard rank intervals / benchmark confidence intervals:** real gap, but crowded in 2026 (several
  preprints plus a NIST report).
- **Quantum error mitigation tooling:** the claimed tooling gap isn't stated in any source I found, and
  there's no hardware here to validate against.
- **Notification-timing RL:** an RL framing already exists, so only a narrow optimal-stopping claim is left.
