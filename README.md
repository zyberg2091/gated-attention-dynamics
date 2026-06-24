# Gated Attention for Language Models

## Abstract

We study a sigmoid gate applied to attention outputs in a causal transformer under different normalization settings. Gating reduces mean activation from 0.50 → 0.257 and induces strong layer-dependent sparsity (Layer 0: 4%, Layers 1–3: 43–54%). Gated models achieve marginally lower perplexity (143.6 vs 145.7) and reduced loss variance. No consistent reduction in attention sink is observed. QK normalization degrades performance across all configurations.

---

## 1. Setup

Model: 4-layer transformer, d_model=256, 8 heads, FFN=1024, RoPE, RMSNorm  
Training: 5 epochs, 500 steps/epoch, batch=16, Adam (1e-3), context=128  

Gate:  
Linear projection → sigmoid → elementwise scaling of attention output  
Init: bias=0 → gate≈0.5  

Sparsity: fraction of gate values < 0.1  

Configs:
- No-gate
- No-gate + QK norm
- Gate
- Gate + QK norm

---

## 2. Results

### 2.1 Gate Dynamics

Gate mean: 0.50 → 0.257  
Gate std: ~0.08 → ~0.22  

Indicates transition from uniform scaling to bimodal distribution.

---

### 2.2 Sparsity

Final sparsity: 0.369  

Growth is front-loaded (epoch 0 contributes ~25% of final value), followed by plateau.

---

### 2.3 Layer-wise Structure

| Layer | Mean | Sparsity |
|---|---|---|
| 0 | 0.427 | 0.044 |
| 1 | 0.162 | 0.542 |
| 2 | 0.202 | 0.460 |
| 3 | 0.235 | 0.431 |

Layer 0 remains dense.  
Layers 1–3 are sparse (≈43–54%).  

This defines a depth-stratified regime: one dense layer + three sparse layers.

---

### 2.4 Loss / Perplexity

| Epoch | Gated | No-gate |
|---|---|---|
| 1 | 691.8 | 709.3 |
| 5 | 143.6 | 145.7 |

Gated model consistently lower across epochs.  
Variance is reduced (fewer high-loss spikes).

---

### 2.5 Gradient Norm

Grad norm: ~1.3 → ~0.6 (early) → ~0.8–0.88 (late)  
No instability observed.

---

### 2.6 Attention Sink

- Gated: 0.044  
- No-gate: 0.055  

Ranges overlap; no directional trend.

---

## 3. Discussion

### QK Normalization

QK normalization increases final PPL (145.7 → 171.3).  
Suggests reduced expressivity or interference with RoPE.

---

### Interpretation of Gate

Gate functions as head selection:

- low-value heads → suppressed  
- high-value heads → retained  

Observed sparsity (~30–50%) matches known head pruning ranges.

---

## 4. Conclusion

- Gating induces strong depth-dependent sparsity  
- Improves stability and slightly reduces perplexity  
- Does not suppress attention sink under current measurement  

Primary finding: **emergent depth-stratified sparsity (dense early layer, sparse deeper layers)**.

**Inspired by:**  
[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/pdf/2505.06708)

This work is inspired by the use of sigmoid gating to introduce non-linearity and sparsity in attention. Unlike the original paper, which applies head-wise gating, we apply element-wise gating over the attention output dimensions. This modification enables finer-grained suppression but may alter the expected impact on attention sink behavior.
