import sys, numpy as np, jax, jax.numpy as jnp, pydoc_data.topics as t
from gpt2 import *
import os; D = os.environ.get("GPT2_DIR", "gpt2")  # dir with model.safetensors + tokenizer.json (openai-community/gpt2)
p, tok = load(D)
text = " ".join(t.topics.values())[:20000]
ids = np.array(tok.encode(text).ids[:513])[None]
f = jax.jit(hidden, static_argnums=2)
for name, dt in [("fp32", jnp.float32), ("bf16", jnp.bfloat16), ("fp16", jnp.float16)]:
    h = f(p, ids[:, :-1], dt); lg = logits(p, h)
    nll = -jax.nn.log_softmax(lg, -1)[0, np.arange(512), ids[0, 1:]].mean()
    print(name, "nll", float(nll), "tokens", 512, "h dtype", h.dtype, flush=True)
