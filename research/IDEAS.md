# Research-gap ideas (2026-11-06)

Searched for gaps where a small piece of math plus software could close them. Each idea is pitched in the
"category gap" shape: *tools measure X, but what matters is Y.*

**How sure am I?** The gaps come from web-search snippets of 2026 preprints. I did not read the full papers, and
my novelty checks were a few searches each, not a literature review. Verify before claiming novelty.

---

## 1. "Safe token" certificates for LLM output  — researched on real GPT-2 (see margin_certificate/README.md)

**One line:** The same prompt gives different answers on different GPUs. Which tokens are safe, in advance, with a guarantee?

**Correction to my first pitch:** I said no per-token bound existed. A generic margin lemma (margin > 2x logit
error => same argmax) already exists, so that part is not new. And on real GPT-2 my per-token "cancellation"
error term barely beat the plain error budget (+2.5 points certified, sd 0.4), and the raw margin predicts
flips as well as any normalised score. The synthetic prototype's good news did not transfer: a Gaussian
flip-probability model predicted 1.5% flips vs 3.9% actual.

**What survived:** greedy tokens differ bf16-vs-fp16 at 3.97% of positions on real GPT-2, and a single
conformally-calibrated margin threshold gives a finite-sample guarantee: at alpha=1%, margin > 0.158 nats
certifies 91% of tokens; at alpha=0.1%, margin > 0.366 nats certifies 81%. That is a useful, deployable,
honest result, but it is closer to "calibrated risk control for a known phenomenon" than a new theorem.

**Still open (and where novelty might be):** sequence-level divergence prediction under free-running
generation, domain shift of the calibration, larger models, other kernel/hardware pairs. None tested yet.

Sources: [cross-GPU nondeterminism](https://arxiv.org/abs/2609.25624),
[greedy decoding not precision-invariant](https://arxiv.org/pdf/2609.26621),
[margin-triggered verification](https://arxiv.org/html/2605.30218v1),
[Lipschitz certification under floating point](https://arxiv.org/abs/2603.13334).

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
