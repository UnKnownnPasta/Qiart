"""GPT-2 small forward pass in JAX with a selectable working precision (fp32 / bf16 / fp16).
Everything in the body (LayerNorm, attention, softmax, GELU, residuals) runs in the working dtype;
the lm_head accumulates in fp32 (as tensor-core kernels do). Weights are cast once from the fp32 checkpoint."""
import json, numpy as np, jax, jax.numpy as jnp
from safetensors.numpy import load_file
from tokenizers import Tokenizer

def load(dirpath):
    raw = load_file(f"{dirpath}/model.safetensors")
    raw = {k.removeprefix("transformer."): v for k, v in raw.items()}
    L = 12
    p = {"wte": raw["wte.weight"], "wpe": raw["wpe.weight"], "lnf_g": raw["ln_f.weight"], "lnf_b": raw["ln_f.bias"], "blocks": []}
    for i in range(L):
        g = lambda n: raw[f"h.{i}.{n}"]
        p["blocks"].append({k: g(v) for k, v in dict(
            ln1_g="ln_1.weight", ln1_b="ln_1.bias", ln2_g="ln_2.weight", ln2_b="ln_2.bias",
            qkv_w="attn.c_attn.weight", qkv_b="attn.c_attn.bias", o_w="attn.c_proj.weight", o_b="attn.c_proj.bias",
            up_w="mlp.c_fc.weight", up_b="mlp.c_fc.bias", dn_w="mlp.c_proj.weight", dn_b="mlp.c_proj.bias").items()})
    return p, Tokenizer.from_file(f"{dirpath}/tokenizer.json")

def _ln(x, g, b):
    dt = x.dtype; x = x.astype(jnp.float32)          # fused-kernel style: fp32 statistics, output in working dtype
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) * jax.lax.rsqrt(v + 1e-5) * g.astype(jnp.float32) + b.astype(jnp.float32)
    return y.astype(dt)

def hidden(p, toks, dtype=jnp.float32, H=12):
    c = lambda t: jax.tree_util.tree_map(lambda a: jnp.asarray(a).astype(dtype), t)
    p = c(p); B, T = toks.shape; d = p["wte"].shape[1]
    x = p["wte"][toks] + p["wpe"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for b in p["blocks"]:
        q, k, v = jnp.split(_ln(x, b["ln1_g"], b["ln1_b"]) @ b["qkv_w"] + b["qkv_b"], 3, -1)
        sh = lambda a: a.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(0, 1, 3, 2)).astype(jnp.float32) * (d // H) ** -.5   # scores rounded to dtype by the matmul, scaled in fp32
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), -1).astype(dtype)
        x = x + (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ b["o_w"] + b["o_b"]
        x = x + jax.nn.gelu((_ln(x, b["ln2_g"], b["ln2_b"]) @ b["up_w"] + b["up_b"]).astype(jnp.float32), approximate=True).astype(dtype) @ b["dn_w"] + b["dn_b"]
    return _ln(x, p["lnf_g"], p["lnf_b"])          # (B,T,d) in `dtype`

def logits(p, h):
    """fp32-accumulated lm_head (tied embeddings) applied to hidden state h (any dtype)."""
    W = jnp.asarray(p["wte"]).astype(h.dtype).astype(jnp.float32)
    return h.astype(jnp.float32) @ W.T
