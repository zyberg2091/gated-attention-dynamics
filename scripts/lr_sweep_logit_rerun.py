"""
c1_lr_saturation_test.py

Question: how does C1's high-LR boundary-heavy gate differ from its lower-LR
suppressor regime?

Two tests in one run:
  TEST 1 (LR sensitivity): train C1 at a lower LR.
          If the gate returns to a suppressor (open ~0%), the high-LR
          boundary mass is strongly learning-rate dependent.
  TEST 2 (logit magnitude): print mean |gate logit| for C1@3e-3 vs C3@3e-3.
          If C1's is much larger, its gate is more severely saturated.

Each line reports:  val_loss | open% | closed% | mean gate | mean|logit|
  - val_loss   : final-batch diagnostic only; not used in the conclusion
  - open%      : gate values > 0.99
  - closed%    : gate values < 0.01
  - mean|logit|: average magnitude of the pre-sigmoid gate value. Big = saturated.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random, math, itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# RoPE (verbatim)
def compute_rope_params(seq_len, head_dim):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = (torch.arange(seq_len).float()[:, None] * theta[None, :])[None, None, :, :]
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin, offset=0):
    b, h, t, d = x.shape
    cs = cos[..., offset:offset + t, :].to(x.device, x.dtype)
    sn = sin[..., offset:offset + t, :].to(x.device, x.dtype)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cs - xo * sn
    out[..., 1::2] = xe * sn + xo * cs
    return out


# Model
class Block(nn.Module):
    def __init__(self, d, ctx, dropout, ff, heads, qk_norm, pre_norm, post_norm):
        super().__init__()
        self.heads, self.hd, self.ctx = heads, d // heads, ctx
        self.wq = nn.Linear(d, d, bias=False); self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False); self.out_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.norm1 = nn.RMSNorm(d); self.norm2 = nn.RMSNorm(d)
        self.gate = nn.Linear(d, d)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 2.0)
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), diagonal=1))
        self.qk_norm, self.pre_norm, self.post_norm = qk_norm, pre_norm, post_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.hd); self.k_norm = nn.RMSNorm(self.hd)
        if pre_norm and post_norm:
            self.post_attn_norm = nn.RMSNorm(d)
            self.post_ff_norm = nn.RMSNorm(d); self.pre_ff_norm = nn.RMSNorm(d)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        xn = self.norm1(x) if self.pre_norm else x
        q = self.wq(xn).view(b, t, self.heads, self.hd).transpose(1, 2)
        k = self.wk(xn).view(b, t, self.heads, self.hd).transpose(1, 2)
        v = self.wv(xn).view(b, t, self.heads, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(2, 3)) / (self.hd ** 0.5)
        att = att.masked_fill(self.mask[:t, :t].bool(), float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        z = att @ v
        gate = self.gate(xn).view(b, t, self.heads, self.hd).transpose(1, 2)   # pre-sigmoid logit
        z = z * torch.sigmoid(gate)
        z = z.transpose(1, 2).contiguous().view(b, t, -1)
        attn_out = self.out_proj(z)
        if self.pre_norm and self.post_norm:                 # C3
            attn_out = self.post_attn_norm(attn_out)
            x = x + attn_out
            x = x + self.post_ff_norm(self.ff(self.pre_ff_norm(x)))
        elif self.pre_norm:                                  # C1 / C2
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
        return x, gate

class LM(nn.Module):
    def __init__(self, d, ctx, dropout, ff, heads, vocab, qk_norm, pre_norm, post_norm, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, ctx, dropout, ff, heads, qk_norm, pre_norm, post_norm)
                                     for _ in range(layers)])
        cos, sin = compute_rope_params(ctx, d // heads)
        self.register_buffer("cos", cos); self.register_buffer("sin", sin)
        self.final_norm = nn.RMSNorm(d); self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        gates = []
        x = self.emb(x)
        for blk in self.blocks:
            x, g = blk(x, self.cos, self.sin)
            gates.append(g)
        return self.head(self.final_norm(x)), gates


# one run: train at a given LR, return loss + gate stats + logit magnitude 
def run(dataset, vocab, qk, pre, post, lr, seed=12, epochs=10, layers=8):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_val = int(len(dataset) * 0.2)
    tr, va = random_split(dataset, [len(dataset) - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=16, shuffle=True, generator=gen)
    vl = DataLoader(va, batch_size=16, shuffle=False)

    model = LM(256, 128, 0.1, 1024, 8, vocab, qk, pre, post, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(xb)
            loss = lossf(out.view(-1, vocab), yb.view(-1))
            if not torch.isfinite(loss):
                print("    diverged (non-finite loss)", flush=True)
                return {"val_loss": float("nan"), "open": float("nan"),
                        "closed": float("nan"), "mean": float("nan"), "logit": float("nan")}
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"    epoch {ep+1}/{epochs} train_loss={loss.item():.3f}", flush=True)

    # eval: clean val loss + gate stats + gate-logit magnitude
    model.eval()
    vloss, vn = 0.0, 0
    o = c = m = lg = n = 0.0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(vl):
            if i >= 50: break
            xb, yb = xb.to(device), yb.to(device)
            out, gates = model(xb)
            vloss += lossf(out.view(-1, vocab), yb.view(-1)).item(); vn += 1
            logits = torch.stack(gates)              # pre-sigmoid
            g = torch.sigmoid(logits)                # actual gate values
            o  += (g > 0.99).float().mean().item()
            c  += (g < 0.01).float().mean().item()
            m  += g.mean().item()
            lg += logits.abs().mean().item()         # blow-up fingerprint
            n  += 1
    return {"val_loss": vloss / vn, "open": 100 * o / n, "closed": 100 * c / n,
            "mean": m / n, "logit": lg / n}


#data prep (verbatim, bounded)
def build_dataset(max_samples, seq_len=128, batch=50000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    print("loading + tokenizing (only as much as needed)...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    texts = [t for t in ds["train"]["text"] if len(t.strip()) > 0]
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    eos = tok.eos_token_id
    need = max_samples * seq_len + 1
    chunks, total = [], 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch])["input_ids"]
        for s in enc:
            s.append(eos)
        arr = np.fromiter(itertools.chain.from_iterable(enc), dtype=np.int64)
        chunks.append(arr); total += len(arr)
        print(f"  {i + batch:>8} texts -> {total/1e6:5.1f}M tokens", flush=True)
        if total >= need:
            break
    toks = torch.from_numpy(np.concatenate(chunks))

    class WT(Dataset):
        def __init__(self, t, L): self.t, self.L = t, L; self.n = (len(t) - (L + 1)) // L
        def __len__(self): return self.n
        def __getitem__(self, i): ch = self.t[i*self.L:i*self.L + self.L + 1]; return ch[:-1], ch[1:]

    full = WT(toks, seq_len)
    return Subset(full, range(min(max_samples, len(full)))), tok.vocab_size


if __name__ == "__main__":
    MAX_SAMPLES = 100000   # same budget as LR run, so numbers are comparable

    data, vocab = build_dataset(MAX_SAMPLES)
    print(f"samples={len(data)} vocab={vocab} device={device}\n", flush=True)

    # (name, qk_norm, pre_norm, post_norm, lr)
    RUNS = [
        ("C1 @ 5e-4  (lower LR: should train well, expect suppressor)", False, True, False, 5e-4),
        ("C1 @ 1e-3  (baseline suppressor anchor)",                     False, True, False, 1e-3),
        ("C1 @ 3e-3  (broken: high loss, saturated gate)",             False, True, False, 3e-3),
        ("C3 @ 3e-3  (stable reference for logit magnitude)",           True,  True, True,  3e-3),
    ]

    out = []
    for name, qk, pre, post, lr in RUNS:
        print(name, flush=True)
        r = run(data, vocab, qk, pre, post, lr)
        out.append((name, r))
        print(f"  -> val_loss={r['val_loss']:.2f}  open={r['open']:.2f}%  "
              f"closed={r['closed']:.2f}%  mean={r['mean']:.3f}  |logit|={r['logit']:.2f}\n", flush=True)

    print("\n==================== SUMMARY ====================")
    print(f"{'run':<58}{'val_loss':>9}{'open':>8}{'closed':>9}{'|logit|':>9}")
    for name, r in out:
        print(f"{name:<58}{r['val_loss']:>9.2f}{r['open']:>7.2f}%{r['closed']:>8.2f}%{r['logit']:>9.2f}")

    #logs
    d = {name.split()[0] + "@" + name.split()[2]: r for name, r in out}
    print("\n-------------------- READING --------------------")
    print("TEST 1 (did the suppressor return at a healthy LR?)")
    print("  Compare C1 @ 5e-4 open%  vs  C1 @ 3e-3 open%.")
    print("  If 5e-4 open% is near 0 AND its val_loss is ~3.3, the high-LR")
    print("  extremes were instability, not learned selection.\n")
    print("TEST 2 (is C1's high-LR gate saturated from blow-up?)")
    print("  Compare |logit| for C1 @ 3e-3  vs  C3 @ 3e-3.")
    print("  If C1's |logit| is much larger, its gate hit the rails because")
    print("  the logits exploded, not because it learned a bimodal code.")
