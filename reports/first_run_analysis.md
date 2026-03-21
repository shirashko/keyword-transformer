# First Run Analysis

> **Date:** 2026-03-21 | **Branch:** `feat/update-scripts-and-readme` | **Checkpoints:** `models_data_v2_12_labels/first_run_on_server/`

---

## TL;DR

| Model | Val Accuracy | Test Accuracy | Paper Target |
|:------|:-------------|:--------------|:-------------|
| Baseline KWT-1 | 96.88% | **97.17%** | 97.72% |
| Distillation KWT-1 | **100%** | **96.35%** | 98.08% |

The distillation model's perfect validation score is misleading. On the held-out test set, it actually performs **worse** than the baseline. The root cause is a code change that replaced **AdamW** (with weight decay) with **plain Adam** (no regularization), compounded by several other deviations from the paper's setup.

---

## 1. What the Paper Prescribes

All KWT experiments in Berg et al. share a single set of hyperparameters (Table 2):

| Parameter | Value |
|:----------|:------|
| Optimizer | AdamW |
| Weight decay | 0.1 |
| Learning rate | 0.001 |
| Batch size | 512 |
| Steps | 23,000 |
| Warmup epochs | 10 |
| Label smoothing | 0.1 |
| LR schedule | Cosine |

---

## 2. What We Ran

### Baseline (`scripts/train_baseline.slurm`)

| Parameter | Ours | Paper | Issue |
|:----------|:-----|:------|:------|
| Optimizer | Adam | AdamW | **Bug** -- weight decay removed |
| Weight decay | 0 (code ignores flag) | 0.1 | **Bug** |
| Learning rate | 0.0005 | 0.001 | Half |
| Batch size | 256 | 512 | Half |
| Steps | 46,876 | 23,000 | Scaled for batch size (same 12M examples) |
| Warmup epochs | 10 | 10 | OK |
| Effective LR | 0.00025 | 0.001 | 4x lower (LR is scaled by `batch/512`) |

### Distillation (`scripts/train_distill.slurm`)

| Parameter | Ours | Paper | Issue |
|:----------|:-----|:------|:------|
| Optimizer | Adam | AdamW | **Bug** -- same as baseline |
| Weight decay | 0 | 0.1 | **Bug** -- also passed `--l2_weight_decay 0.0` |
| Learning rate | 0.0005 | 0.001 | Half |
| Batch size | **100** | 512 | **5x smaller** (inherited from upstream ARM repo) |
| Steps | 120,000 | 23,000 | Scaled for batch size (same 12M examples) |
| Warmup epochs | 5 | 10 | Half |
| Effective LR | **0.0001** | 0.001 | **10x lower** |
| Gradient updates | **120k** | 23k | **5x more** weight updates |

> The `batch_size=100` for distillation was copied from the upstream ARM repo's `distill.sh` and the default in `model_params.py:74`. It contradicts the paper.

---

## 3. Test Set Results

We evaluated both checkpoints on the **test set** (unseen during both training and validation). Two independent eval methods produced identical results.

| Model | Val Accuracy | Test Accuracy | Paper |
|:------|:-------------|:--------------|:------|
| Baseline | 96.88% | **97.17%** | 97.72% |
| Distillation | **100%** | **96.35%** | 98.08% |

**The distillation model with 100% val accuracy scores 0.8% below the baseline on the test set.** The baseline (97.17%) is close to the paper's 97.72% despite missing weight decay.

### How to reproduce

```bash
source venv3/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export TF_USE_LEGACY_KERAS=1

# Baseline
python eval_checkpoint.py \
  ./models_data_v2_12_labels/first_run_on_server/kwt1_baseline/ \
  ./data2/speech_commands_v0.02/ \
  --mel_upper_edge_hertz 7600 --mel_num_bins 80 --dct_num_features 40 \
  --window_size_ms 30.0 --window_stride_ms 10.0 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"

# Distillation
python eval_checkpoint.py \
  --distill ./distill_att_mh_rnn.json \
  ./models_data_v2_12_labels/first_run_on_server/kwt1_distill/ \
  ./data2/speech_commands_v0.02/ \
  --mel_upper_edge_hertz 7600 --mel_num_bins 80 --dct_num_features 40 \
  --window_size_ms 30.0 --window_stride_ms 10.0 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"
```

> The dataset (~2.3GB) auto-downloads on first run. Audio feature flags must be passed explicitly -- defaults don't match the training config.

---

## 4. Why 100% Val but Only 96% Test?

The model never trains on validation data, so this isn't classical overfitting. Instead, several factors combine to produce overconfident predictions that happen to be correct on the fixed validation set but don't fully generalize.

### No weight decay

Both runs use plain Adam. Without weight decay, weights grow freely, making predictions sharper over time. The baseline survives because it trains for fewer steps.

### 2.5x more gradient updates

Both runs see the same 12M training examples, but the distillation model takes 120k gradient steps vs 47k. Each step without weight decay lets weights grow further.

### Very low effective learning rate

The LR is scaled by `batch_size / 512`. With `batch_size=100`, the distillation model's peak LR is only **0.0001** -- ten times lower than the paper's 0.001. This allows the model to slowly and precisely fit the clean data distribution over 120k steps.

### Augmentation gap

Training uses heavy augmentation (SpecAugment, time shifting, resampling, background noise). Validation and test use none. The model learns sharp decision boundaries that classify clean audio perfectly but break under perturbation -- explaining the 84% train accuracy alongside 100% val accuracy.

### Ensemble smoothing

Val accuracy reports the **ensemble** (average of class + distill token logits), which is more robust than either head alone. This gives a small accuracy boost on the clean val set.

---

## 5. Fixes for Next Run

### Code fix -- restore AdamW in `train.py`

```python
elif flags.optimizer == 'adamw':
    optimizer = tfa.optimizers.AdamW(
        learning_rate=float(flags.learning_rate),
        weight_decay=flags.l2_weight_decay
    )
```

### Script fixes -- match the paper

```diff
  # Distillation script
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

### Checkpoint loading fix -- `test.py`

The branch changed saving to `tf.train.Saver` (name-based format) but `test.py` still used `model.load_weights()` (object-based), breaking the eval pipeline. All 5 `load_weights()` calls were updated to use `saver.restore()` and the default `weights_name` changed from `'best_weights'` to `'best_weights.ckpt'`.

---

## 6. Local Environment Setup

```bash
# TF 2.4 doesn't install on Python 3.10/macOS -- use TF 2.15 (matches server)
python3.10 -m venv venv3
source venv3/bin/activate
pip install "tensorflow==2.15.*" "tf-keras==2.15.0" tensorflow_addons \
    pydot graphviz "numpy<2" absl-py transformers
```

The dataset auto-downloads to `data2/speech_commands_v0.02/` (~2.3GB).
