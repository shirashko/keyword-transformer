# First Run Analysis Report

**Date:** 2026-03-21
**Branch:** `feat/update-scripts-and-readme`
**Runs analyzed:** `models_data_v2_12_labels/first_run_on_server/kwt1_baseline/` and `kwt1_distill/`

---

## 1. Summary

The first training run on the university server produced a **baseline KWT-1** model and a **distillation KWT-1** model. The distillation model showed 100% validation accuracy, which raised concerns. After investigation, we found:

- The optimizer was changed from **AdamW** (with weight decay) to **plain Adam** (no weight decay) — affecting both runs.
- The distillation model's 100% validation accuracy was misleading — on the **test set** it scored **96.10%**, worse than the baseline's **97.27%**.
- Multiple compounding factors contributed to the inflated validation accuracy.

---

## 2. Paper Reference (Table 2 — Hyperparameters)

The KWT paper (Berg et al.) specifies a single set of hyperparameters for **all** experiments (baseline and distillation):

| Parameter | Paper Value |
|---|---|
| Batch size | **512** |
| Training steps | **23,000** |
| Optimizer | **AdamW** |
| Learning rate | **0.001** |
| LR schedule | Cosine |
| Warmup epochs | 10 |
| Weight decay | **0.1** |
| Label smoothing | 0.1 |
| Dropout | 0 |

Paper results for KWT-1 on V2-12:
- **Baseline:** 97.72%
- **With distillation:** 98.08%

---

## 3. What We Ran vs. What the Paper Specifies

### 3.1 Baseline (`scripts/train_baseline.slurm`)

| Parameter | Our Value | Paper Value | Match? |
|---|---|---|---|
| Batch size | 256 | 512 | No (half) |
| Training steps | 46,876 | 23,000 | Adjusted (same 12M total examples) |
| Optimizer | **Adam** (bug) | AdamW | **NO** |
| Learning rate | 0.0005 | 0.001 | No (half) |
| Weight decay | 0.1 (flag) / **0 (actual)** | 0.1 | **NO (ignored by code)** |
| Warmup epochs | 10 | 10 | Yes |
| Effective LR (after batch scaling) | 0.00025 | 0.001 | No (4x lower) |

### 3.2 Distillation (`scripts/train_distill.slurm`)

| Parameter | Our Value | Paper Value | Match? |
|---|---|---|---|
| Batch size | **100** | 512 | **No (5x smaller)** |
| Training steps | 120,000 | 23,000 | Adjusted (same 12M total examples) |
| Optimizer | **Adam** (bug) | AdamW | **NO** |
| Learning rate | 0.0005 | 0.001 | No (half) |
| Weight decay | **0.0** (flag) / **0 (actual)** | 0.1 | **NO** |
| Warmup epochs | 5 | 10 | No (half) |
| Effective LR (after batch scaling) | **0.0000977** | 0.001 | **No (10x lower)** |

### 3.3 Key Differences Between Our Baseline and Distillation Scripts

| | Baseline | Distillation |
|---|---|---|
| Batch size | 256 | 100 |
| Steps | 46,876 | 120,000 |
| Gradient updates | 46.8k | **120k (2.5x more)** |
| `--l2_weight_decay` flag | 0.1 | 0.0 |
| `--warmup_epochs` | 10 | 5 |
| Effective peak LR | 0.00025 | 0.0000977 |
| `--eval_step_interval` | 144 | 500 |
| Total examples seen | ~12M | ~12M |

> **Note:** The `batch_size=100` for distillation was inherited from the original ARM upstream repo (`distill.sh`), which also used `--l2_weight_decay 0.`. These values contradict the paper. The default in `kws_streaming/models/model_params.py:74` is also `batch_size = 100`.

---

## 4. The AdamW vs Adam Bug

### What the code was (master branch):

```python
# kws_streaming/train/train.py (master)
from transformers import AdamWeightDecay

elif flags.optimizer == 'adamw':
    exclude = ["pos_emb", "class_emb", "layer_normalization", "bias"]
    optimizer = AdamWeightDecay(
        learning_rate=0.05,  # NOTE: hardcoded, ignores flags.learning_rate
        weight_decay_rate=flags.l2_weight_decay,
        exclude_from_weight_decay=exclude
    )
```

### What the code is now (this branch):

```python
# kws_streaming/train/train.py (feat/update-scripts-and-readme)
elif flags.optimizer == 'adamw':
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(flags.learning_rate))
```

**Both runs pass `--optimizer "adamw"`**, so both hit this code path. The result is **plain Adam with zero weight decay** for both baseline and distillation.

### What AdamW does differently from Adam

- **Adam**: Updates weights using adaptive learning rates based on gradient moments. No weight regularization.
- **AdamW**: Same as Adam, but additionally **shrinks weights by a factor of `weight_decay * lr` every step**. This keeps weights small and prevents overconfident predictions.

The paper specifically notes that increasing weight decay from 0.05 to **0.1** was "important" for keyword spotting.

> **Note:** The master branch code also had a bug: `learning_rate=0.05` was hardcoded instead of using `flags.learning_rate`. Any fix should address both issues.

---

## 5. Training Curves Analysis (TensorBoard)

### 5.1 Loss Graph

| Run | Smoothed | Final Value | Steps |
|---|---|---|---|
| kwt1_baseline / train | 0.9072 | 0.7667 | 46,876 |
| kwt1_baseline / validation | 0.6008 | 0.6008 | 46,876 |
| kwt1_distill / train | 0.833 | 0.847 | 120,000 |
| kwt1_distill / validation | 0.5434 | 0.5435 | 120,000 |

- Baseline train loss (cyan): Noisy but trending down. Normal.
- Baseline val loss (pink): Smooth convergence to 0.60. Healthy.
- Distill train loss (dark gray): Very noisy, barely decreasing after ~40k steps.
- Distill val loss (orange): Converges to **0.5435**, which is near the theoretical minimum with `label_smoothing=0.1` (~0.526 for 12 classes). This means the model predicts every validation sample with ~95% confidence on the correct class.

### 5.2 Accuracy Graph

| Run | Smoothed | Final Value | Steps |
|---|---|---|---|
| kwt1_baseline / train | 0.835 | 0.8945 | 46,876 |
| kwt1_baseline / validation | 0.9687 | 0.9688 | 46,876 |
| kwt1_distill / train | 0.8559 | 0.84 | 120,000 |
| kwt1_distill / validation | **1.0** | **1.0** | 120,000 |

- Distillation validation accuracy reached **100%**, which is higher than any published result (paper's best: 98.56% with KWT-3).
- Distillation train accuracy (84%) is **lower** than baseline train accuracy (89%), despite training for 2.5x more steps.

### 5.3 Important: Metric Asymmetry

The TensorBoard metrics compare different things for train vs validation in the distillation model:

| | What's measured | Data type |
|---|---|---|
| **Train accuracy** | Single class head (`acc_label`) | Augmented data |
| **Val accuracy** | Ensemble of both heads (`acc_ensemble`) | Clean data |

This asymmetry is defined in `kws_streaming/train/train.py`:
- Train logging (line 211): `tag='accuracy', simple_value=acc_label`
- Val logging (line 251): `tag='accuracy', simple_value=acc_ensemble`

---

## 6. Test Set Evaluation

We evaluated both checkpoints on the **test set** (never seen during training or validation) using a custom eval script.

### 6.1 Results

| Model | Val Accuracy | **Test Accuracy** | Paper (KWT-1) |
|---|---|---|---|
| Baseline | 96.88% | **97.27%** | 97.72% |
| Distillation (ensemble) | 100% | **96.10%** | 98.08% |
| Distillation (label head only) | — | 96.08% | — |
| Distillation (distill head only) | — | 96.15% | — |

**The distillation model with 100% validation accuracy performs worse than the baseline on the test set.**

The baseline result (97.27%) is reasonably close to the paper's 97.72%, despite missing weight decay. The distillation model (96.10%) significantly underperforms, ~2% below the paper's 98.08%.

### 6.2 Eval Script

The standard eval code (`kws_streaming/train/test.py`) uses `model.load_weights()` which is incompatible with the `tf.train.Saver` checkpoint format used by the current branch. We wrote a custom eval script (`eval_checkpoint.py`) that loads checkpoints via `tf.train.Saver`.

```bash
# Setup
source venv3/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export TF_USE_LEGACY_KERAS=1

# Baseline eval
python eval_checkpoint.py \
  ./models_data_v2_12_labels/first_run_on_server/kwt1_baseline/ \
  ./data2/speech_commands_v0.02/ \
  --mel_upper_edge_hertz 7600 --mel_num_bins 80 --dct_num_features 40 \
  --window_size_ms 30.0 --window_stride_ms 10.0 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"

# Distillation eval (requires --distill flag to build model with distillation token)
python eval_checkpoint.py \
  --distill ./distill_att_mh_rnn.json \
  ./models_data_v2_12_labels/first_run_on_server/kwt1_distill/ \
  ./data2/speech_commands_v0.02/ \
  --mel_upper_edge_hertz 7600 --mel_num_bins 80 --dct_num_features 40 \
  --window_size_ms 30.0 --window_stride_ms 10.0 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"
```

> **Note:** The dataset (~2.3GB) auto-downloads on first run if not present locally. Audio feature flags (`--mel_num_bins 80 --dct_num_features 40` etc.) must be passed explicitly — the defaults don't match the training config.

> **Note:** The eval code in `test.py` uses `model.load_weights()` (object-based checkpoints), but the branch changed saving to `tf.train.Saver` (name-based checkpoints). These are incompatible. The custom `eval_checkpoint.py` works around this by also using `tf.train.Saver` for loading.

---

## 7. Root Cause Analysis — Why 100% Val Accuracy?

The 100% validation accuracy is **real** (the model does classify all ~9,900 validation samples correctly) but **misleading** (it doesn't generalize — test accuracy is only 96.10%).

Contributing factors, ordered by importance:

### 7.1 No weight decay (Adam instead of AdamW)

Both runs use plain Adam. Without weight decay, weights grow unbounded, making predictions increasingly sharp/overconfident. The baseline survives because it trains for fewer steps. The paper's `weight_decay=0.1` is the primary regularizer.

### 7.2 2.5x more gradient updates (120k vs 47k)

Same total data (12M examples), but the distillation model does 120k weight updates vs 47k for baseline. Without weight decay to counteract this, the model overfits more with each additional update.

### 7.3 Very low effective learning rate

The cosine LR schedule scales by `batch_size / 512`:
- Baseline: `0.0005 * 256/512 = 0.00025`
- Distillation: `0.0005 * 100/512 = 0.0000977` (10x below paper's 0.001)

The very low LR allows the model to slowly and precisely fit the clean data distribution.

### 7.4 Distillation provides an easier learning signal

The teacher (MHAtt-RNN, ~98.4% accurate) reinforces correct predictions. Both heads converge faster, leaving more steps for overfitting.

### 7.5 Augmentation asymmetry

Heavy augmentation during training (SpecAugment, time shift, resampling, noise) but none during validation. The model learns sharp decision boundaries that work on clean data but break under augmentation. This is not "overfitting to validation" — the model never trains on validation data — but it produces misleadingly high validation accuracy.

### 7.6 Ensemble metric on validation

Validation reports the **ensemble** of two heads on clean data, while training reports a **single head** on augmented data. This inflates the apparent gap.

---

## 8. Fixes for Next Run

1. **Restore AdamW with `weight_decay=0.1`** — Use `tfa.optimizers.AdamW` or restore `transformers.AdamWeightDecay` (fix the hardcoded `lr=0.05` on master).
2. **Increase batch size** — Use 256 or 512 for distillation (not 100). Adjust steps to keep 12M total examples.
3. **Match the paper's learning rate** — Use `0.001`, not `0.0005`.
4. **Use `warmup_epochs=10`** for distillation (not 5).
5. **Set `--l2_weight_decay 0.1`** for distillation (paper uses same weight decay for all experiments).

### Recommended distillation script changes:

```diff
- --batch_size 100
+ --batch_size 256
- --how_many_training_steps "120000"
+ --how_many_training_steps "46876"
- --learning_rate "0.0005"
+ --learning_rate "0.001"
- --l2_weight_decay 0.0
+ --l2_weight_decay 0.1
- --warmup_epochs 5
+ --warmup_epochs 10
```

And in `train.py`, restore proper AdamW:

```python
elif flags.optimizer == 'adamw':
    optimizer = tfa.optimizers.AdamW(
        learning_rate=float(flags.learning_rate),
        weight_decay=flags.l2_weight_decay
    )
```

---

## 9. Environment Setup (Local Eval)

To run evaluation locally (macOS):

```bash
# TF 2.4 doesn't install on Python 3.10/macOS — use TF 2.15 (matches server)
python3.10 -m venv venv3
source venv3/bin/activate
pip install "tensorflow==2.15.*" "tf-keras==2.15.0" tensorflow_addons \
    pydot graphviz "numpy<2" absl-py transformers
```

The dataset auto-downloads to `data2/speech_commands_v0.02/` (~2.3GB).
