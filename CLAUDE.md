# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAU final project reproducing the **Keyword Transformer (KWT)** paper (Axel Berg et al.) for 12-label keyword spotting on Google Speech Commands V2. Compares baseline KWT-1 training vs. knowledge distillation from an Att-MH-RNN teacher model.

- **Python 3.10**, **TensorFlow 2.4** (legacy Keras mode: `TF_USE_LEGACY_KERAS=1`)
- Dataset: Google Speech Commands V2 (12 keywords: yes, no, up, down, left, right, on, off, stop, go + silence + unknown)

## Commands

### Environment Setup
```bash
python3.10 -m venv venv3 && source venv3/bin/activate
pip install -r requirements.txt
```

### Training (run from project root, requires `PYTHONPATH` set to project root)
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export TF_USE_LEGACY_KERAS=1

# Baseline KWT-1
python -m kws_streaming.train.model_train_eval \
  --data_dir ./data2/speech_commands_v0.02/ \
  --train_dir ./models_data_v2_12_labels/kwt1_baseline/ \
  --train 1 --batch_size 256 --optimizer adamw --lr_schedule cosine \
  --learning_rate 0.0005 --how_many_training_steps 46876 \
  --mel_upper_edge_hertz 7600 --window_size_ms 30.0 --window_stride_ms 10.0 \
  --mel_num_bins 80 --dct_num_features 40 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"

# Distillation (requires teacher weights in models_data_v2_12_labels/att_mh_rnn_1/)
python -m kws_streaming.train.model_train_eval \
  --data_dir ./data2/speech_commands_v0.02/ \
  --train_dir ./models_data_v2_12_labels/kwt1_distill/ \
  --train 1 --batch_size 100 --optimizer adamw --lr_schedule cosine \
  --learning_rate 0.0005 --how_many_training_steps 120000 \
  --mel_upper_edge_hertz 7600 --window_size_ms 30.0 --window_stride_ms 10.0 \
  --mel_num_bins 80 --dct_num_features 40 \
  --distill_teacher_json ./distill_att_mh_rnn.json \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"
```

### Evaluation
```bash
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

### TensorBoard
```bash
tensorboard --logdir ./models_data_v2_12_labels/first_run_on_server/
```

### Tests
```bash
python -m kws_streaming.data.input_data_test
```

## Architecture

### Entry Point & CLI
`kws_streaming/train/model_train_eval.py` — Uses argparse with subparsers. Base flags (data, audio features, training hyperparams) are in `kws_streaming/train/base_parser.py`. Each model registers its own flags via `model_parameters()`. The model name subcommand (e.g. `kws_transformer`) selects both the parser and model builder.

### Training Pipeline
`kws_streaming/train/train.py` — Training loop with:
- AdamW optimizer, cosine LR schedule, SpecAugment
- **Distillation mode**: activated by `--distill_teacher_json` flag pointing to a JSON config (e.g. `distill_att_mh_rnn.json`). Loads teacher model, adds KL divergence loss + auxiliary average output loss.
- Checkpoints saved per eval step; resume via `--start_step`.

### KWT Model
`kws_streaming/models/kws_transformer.py` — Builds the Keras model. Raw audio → `SpeechFeatures` layer (mel-spectrogram) → `KWSTransformer` (stacked transformer blocks from `transformer_utils.py`) → classification head(s). When distilling, creates 2 output heads + an averaged auxiliary output.

`attention_type` flag controls input representation: `time` (default, attention over time frames), `freq` (over frequency bins), `both` (concatenated), `patch` (ViT-style patches).

### Teacher Model
`kws_streaming/models/att_mh_rnn.py` — CNN + bidirectional GRU + multi-head attention. Config in `distill_att_mh_rnn.json`, pre-trained weights at `models_data_v2_12_labels/att_mh_rnn_1/best_weights`.

### Data Pipeline
`kws_streaming/data/input_data.py` — TF-based data reader. Handles download, train/val/test split, background noise mixing, time shifting, and unknown word sampling.

### Layers Library
`kws_streaming/layers/` — Reusable audio processing layers (mel spectrogram, DCT, MFCC, preemphasis, windowing, SpecAugment). Streaming modes defined in `modes.py`: TRAINING, NON_STREAM_INFERENCE, STREAM_INTERNAL_STATE, STREAM_EXTERNAL_STATE.

### Model Registry
`kws_streaming/train/model_train_eval.py` registers available models (att_mh_rnn, kws_transformer) as subparser commands. Each model module exposes `model_parameters(parser)` and `model(flags)`.

## Key Configuration

### Audio Feature Defaults (for KWT experiments)
- Sample rate: 16kHz, window: 30ms, stride: 10ms
- 80 mel bins, 40 DCT features, mel upper edge: 7600Hz

### KWT-1 Architecture
- 12 transformer layers, 1 attention head, d_model=64, mlp_dim=256, no dropout

### Training Outputs (per run in `train_dir/`)
- `best_weights.*` / `last_weights.*` — checkpoints
- `flags.json` — all hyperparameters for reproducibility
- `logs/train/` and `logs/validation/` — TensorBoard events
- `non_stream/`, `tflite_non_stream/` — exported model formats

## SLURM Scripts
`scripts/train_baseline.slurm` and `scripts/train_distill.slurm` — GPU training scripts for the university cluster. Key differences: distillation uses smaller batch size (100 vs 256), more steps (120k vs 47k), no L2 weight decay.
