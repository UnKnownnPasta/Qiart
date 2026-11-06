"""Run GPT-2 in fp32 / bf16 / fp16 on real prose and cache per-position quantities for the certificate study."""
import sys, numpy as np, jax, jax.numpy as jnp, pydoc_data.topics as t
from gpt2 import *
import os; D = os.environ.get("GPT2_DIR", "gpt2")  # dir with model.safetensors + tokenizer.json (openai-community/gpt2)
NSEQ, T, K = int(sys.argv[1]), 256, 50
p, tok = load(D)
ids = np.array(tok.encode(" ".join(t.topics.values())).ids)
print("corpus tokens", len(ids), flush=True)
starts = np.linspace(0, len(ids) - T - 2, NSEQ).astype(int)          # evenly spread, non-overlapping if corpus is long enough
x = np.stack([ids[s:s + T] for s in starts])
f = jax.jit(hidden, static_argnums=2)
W32 = np.asarray(p["wte"], np.float32)
out = {k: [] for k in ["hA", "hB", "hR", "cand", "zA", "argB", "argR", "seq"]}
for s in range(0, NSEQ, 8):
    xb = x[s:s + 8]
    hs = {n: f(p, xb, dt) for n, dt in [("R", jnp.float32), ("A", jnp.bfloat16), ("B", jnp.float16)]}
    zA, zB, zR = (np.asarray(logits(p, hs[n])).reshape(-1, W32.shape[0]) for n in "ABR")
    cand = np.argsort(-zA, 1)[:, :K]                                     # top-K by implementation A
    out["cand"].append(cand); out["zA"].append(np.take_along_axis(zA, cand, 1))
    out["argB"].append(zB.argmax(1)); out["argR"].append(zR.argmax(1))
    for n in "ABR": out["h" + n].append(np.asarray(hs[n].astype(jnp.float32)).reshape(-1, 768))
    out["seq"].append(np.repeat(np.arange(s, s + len(xb)), T))
    print("batch", s, flush=True)
np.savez("features.npz", **{k: np.concatenate(v) for k, v in out.items()})
