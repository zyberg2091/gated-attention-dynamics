# Gating and Normalization Interact in the Attention Block

Code, notebooks, and raw logs for the paper *Gating and Normalization Interact in the
Attention Block* (under double-blind review; author information withheld).

We train small causal transformers (29–36M parameters) from scratch and study how an
elementwise sigmoid attention-output gate interacts with normalization placement
(pre-norm; pre-norm + QK-norm; sandwich + QK-norm). Every number in the paper's tables
can be traced to a file in this repository; the map is below.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/                      # all notebook experiments ran on Google Colab GPUs
|   ├── main_sweep_wikitext_2.ipynb             # Initial development notebook for the main WikiText-2 sweep
│   ├── main_sweep_54_models_wikitext_2.ipynb   # 54-model factorial sweep (WikiText-2)
│   ├── extended_wikitext_103_L8.ipynb          # 18-model extended run (WikiText-103, L=8)
│   ├── ablation_attn_output_norm.ipynb         # C2+ / C3− single-layer ablation
│   └── param_matched_ffn.ipynb                 # parameter-matched no-gate control
├── scripts/                        # learning-rate analyses (RunPod; see Hardware 
│   ├── lr_sweep_c1_c3.py           # harness 1: C1/C3 across learning rates
│   ├── lr_sweep_logit_rerun.py     # deterministic re-execution adding |logit| + C1@5e-4
│   └── lr_sweep_c1_fine.py         # harness 2: fine-grained C1 stability sweep
│   └── env.txt
├── logs/                           # raw stdout of each script run (names mirror scripts 1:1)
│   ├── lr_sweep_c1_c3.log
│   ├── lr_sweep_logit_rerun.log
│   ├── lr_sweep_c1_fine.log
│   
├── results/
│   └── main_sweep_per_epoch.csv    # per-epoch train/val series for all 54 sweep runs
├── figures/
│   ├── figure_1_tail_ablation.png      # Emergence of gate-value tails during extended training (L=8, WikiText-103)
│   └── figure_2_layer_sparsity.png        # Per-layer gate sparsity (fraction of values below 0.1) 
```

Scripts are intentionally self-contained (the model class is duplicated verbatim in each
file): every log in `logs/` was produced by the script of the same name, and we preserve
that correspondence rather than refactor after the fact.

---

## Paper artifact → repository map

| Paper artifact | File | Where to look |
|---|---|---|
| Tables 1 (main column), 4, 8 (main-sweep rows), 9; Tables 6–7 (Appendix A) | `notebooks/main_sweep_54_models_wikitext_2.ipynb` | The multi-seed loop cell (`SEEDS = [42, 113, 192]` × depths `[4, 8, 12]` × 3 configs × 2 arms = 54 runs). The saved output shows the final session (seed 192) plus the disclosed warm-up cells; the full per-epoch record for all seeds is `results/main_sweep_per_epoch.csv` |
| Tables 1 (extended column), 2, 5, 8 (WT-103 rows), 10; Figures 1 (C1/C2/C3 curves) and 2 | `notebooks/extended_wikitext_103_L8.ipynb` | 18 runs, one banner per (config, arm, seed). Gate statistics are the `Step 14500` log lines (see protocol below) |
| Table 3; Figure 1 (C2+ and C3− curves) | `notebooks/ablation_attn_output_norm.ipynb` | 6 new gated runs (C2+ and C3− × seeds 12/42/100). The C2 and C3 comparison cells are carried over verbatim from `extended_wikitext_103_L8.ipynb` (outputs byte-identical), so Table 3's C2/C3 rows are the extended-run models |
| Table 11 | `notebooks/param_matched_ffn.ipynb` (widened no-gate column) + `extended_wikitext_103_L8.ipynb` (original and gated columns) | 9 widened runs (FFN 1024 → 1152), same recipe and seeds |
| Table 12 | `scripts/lr_sweep_c1_c3.py` → `logs/lr_sweep_c1_c3.log` (five rows) and `scripts/lr_sweep_logit_rerun.py` → `logs/lr_sweep_logit_rerun.log` (the C1@5e-4 row and the entire \|logit\| column) | "Loss" = the minimum end-of-epoch loss line of each run (`loss=` in harness 1's log, `train_loss=` in the rerun) |
| Table 13 | `scripts/lr_sweep_c1_fine.py` → `logs/lr_sweep_c1_fine.log` | All six rows, including the C3 reference |
| First-token attention mass (sink diagnostic) | `Sink Val=` fields in the notebooks' step logs | Logged values span 1.7–8.8% across the four notebooks; the paper reports no dedicated sink evaluation (see its Limitations) |

Rows duplicated across the two LR harnesses (C1@1e-3, C1@3e-3, C3@3e-3) agree within
run-to-run nondeterminism, as stated in the paper's Appendix D.

Figures 1–2 were rendered offline from the logged series (the open/closed-fraction
fields at 500-step intervals in the training logs for Figure 1; the paper's Table 10
per-layer fractions for Figure 2); no plotting script is committed.

---

## Configuration legend (flags used in notebooks and scripts)

| Name | Meaning | Flags |
|---|---|---|
| C1 | pre-norm only | `qk_norm=False, pre_norm=True, post_norm=False` |
| C2 | pre-norm + QK-norm | `qk_norm=True, pre_norm=True, post_norm=False` |
| C3 | sandwich + QK-norm | `qk_norm=True, pre_norm=True, post_norm=True` |
| C2+ | C2 plus **only** the attention-output RMSNorm | C2 flags + `add_norm=True` |
| C3− | C3 minus **only** that layer | C3 flags + `remove_norm=True` |

The attention-output RMSNorm sits after the output projection `W_O`, before the residual
addition. The gate is elementwise, computed from the pre-normalized block input,
initialized with zero weights and bias 2.0 (sigmoid ≈ 0.88).

---

## Measurement protocols (how to read the numbers)

- **Perplexity.** Validation PPL = exp(mean cross-entropy) on the held-out 20% split of the
  official training file (official validation/test files unused).
- **Main sweep (WikiText-2, seq 64, max 6 epochs, patience 2).** Validation PPL reaches
  its minimum at epoch 4 in every one of the 54 runs (verifiable per run in
  `results/main_sweep_per_epoch.csv`); training continues through epoch 6, where the
  patience criterion triggers. Reported main-sweep perplexities (paper Tables 1, 4, 8, 9)
  use the **selected epoch-4 best-validation checkpoint** of each run. The post-optimum
  epoch-6 endpoint and the full per-epoch trajectory are reported separately in the
  paper's Appendix A (Tables 6–7). Both arms use the identical selection rule, so all
  gated-vs-no-gate differences are paired like-for-like.
- **Extended run and ablation (WikiText-103, 300k samples, seq 128, fixed 10 epochs).**
  Early stopping (patience 1) never triggered; all runs use the epoch-10 checkpoint. The
  ablation harness sets `patience=10`, which cannot bind inside a fixed 10-epoch loop —
  identical in effect to the extended recipe.
- **Gate statistics (Tables 2–3).** Computed in training mode on the final logged batch
  (epoch 10, step 14,500), pooled over every token, head, and coordinate in the batch,
  then averaged across seeds. Thresholds: closed < 0.01, open > 0.99, sparsity < 0.1.
- **LR-sweep tables (12–13).** Single seed (12), 100,000-sample budget. "Loss" is the
  minimum end-of-epoch loss value in the log (`loss=` in harness 1's log, `train_loss=`
  in harness 2's log and the rerun). `lr_sweep_logit_rerun.py`
  additionally prints a *validation* loss in its summary and labels it as such; it is not
  the table quantity. `|logit|` is the mean absolute pre-sigmoid gate value over 50
  validation batches, measured in a deterministic re-execution under the fixed seed.

**Known label quirk (does not affect results).** In
`ablation_attn_output_norm.ipynb`, the seed-42/100 banner cells of the C2+ section print
"Config 1 (With Gate)" and the consolidated summary labels those rows "Config 1 (2+)" —
leftover labels. The training calls in those cells carry the C2+ flags
(`post_norm=False, add_norm=True`), and their validation perplexities match Table 3.

---

## Reproducing

```bash
pip install -r requirements.txt

python scripts/lr_sweep_c1_c3.py        # Table 12 rows
python scripts/lr_sweep_logit_rerun.py  # Table 12 |logit| column + C1@5e-4
python scripts/lr_sweep_c1_fine.py      # Table 13
```

Determinism: all scripts set `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=":4096:8"`, and seed 12 for Python/NumPy/PyTorch and the data
split. Each script tokenizes only as much WikiText-103 as its sample budget needs
(~15M tokens) and runs ten epochs per configuration; expect several hours per
script on a single modern GPU.

Notebooks are Colab notebooks; run top-to-bottom. The 54-model sweep checkpoints its
results dictionary to Google Drive (`multi_seed_results_checkpoint.pkl`) so the factorial
can resume across sessions; the exported per-epoch record of that pickle is committed as
`results/main_sweep_per_epoch.csv`.

Datasets (`wikitext-2-raw-v1`, `wikitext-103-raw-v1`) and the GPT-2 tokenizer download
automatically from the Hugging Face Hub on first run.

---

## Hardware and provenance

- **Notebooks** (factorial sweep, extended run, ablation, parameter-matched control):
  single Google Colab GPUs. The notebooks self-attest: Colab metadata blocks, Drive-mount
  outputs, and Colab cache paths appear in the saved outputs. A stale Kaggle metadata
  stamp remains in four notebooks (all but `main_sweep_wikitext_2.ipynb`) from earlier
  editing sessions; all reported runs executed on Colab.
- **Scripts** (learning-rate analyses): RunPod GPU instances across separate sessions.
  `scripts/env.txt` documents the pod used for the deterministic rerun (NVIDIA RTX PRO
  4500 Blackwell); the rerun reproduces harness 1 to the printed precision, so the
  harness-1 numbers are attested on that documented pod. Harness 2 ran in a separate
  session; rows duplicated across the two harnesses agree to the last digits
  (paper Appendix D).

## Data and license

WikiText-2 and WikiText-103 (raw), GPT-2 vocabulary (50,257). Code released under the
LICENSE in this repository. Citation information will be added after the anonymity
period.
