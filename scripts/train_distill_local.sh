#!/bin/bash
# Local distillation training (M4 Mac, ~2 hours)
# Matches paper hyperparameters with batch_size=256 (adjusted steps for same 12M examples)

set -e

KWS_PATH="$(cd "$(dirname "$0")/.." && pwd)"
cd "$KWS_PATH"

source venv3/bin/activate
export PYTHONPATH=$KWS_PATH:$PYTHONPATH
export TF_USE_LEGACY_KERAS=1

DATA_PATH="$KWS_PATH/data2/speech_commands_v0.02"
TRAIN_DIR="$KWS_PATH/models_data_v2_12_labels/kwt1_distill_local/"

rm -rf "$TRAIN_DIR"

echo "Starting KWT-1 Distillation Training (local)..."
echo "Teacher: Att-MH-RNN from distill_att_mh_rnn.json"
echo "Start time: $(date)"

python -m kws_streaming.train.model_train_eval \
  --data_url "" \
  --data_dir "$DATA_PATH/" \
  --train_dir "$TRAIN_DIR" \
  --mel_upper_edge_hertz 7600 \
  --optimizer "adamw" \
  --lr_schedule "cosine" \
  --how_many_training_steps "46876" \
  --eval_step_interval 500 \
  --warmup_epochs 10 \
  --l2_weight_decay 0.1 \
  --learning_rate "0.001" \
  --batch_size 256 \
  --label_smoothing 0.1 \
  --window_size_ms 30.0 \
  --window_stride_ms 10.0 \
  --mel_num_bins 80 \
  --dct_num_features 40 \
  --resample 0.15 \
  --alsologtostderr \
  --train 1 \
  --use_spec_augment 1 \
  --time_masks_number 2 \
  --time_mask_max_size 25 \
  --frequency_masks_number 2 \
  --frequency_mask_max_size 7 \
  --pick_deterministically 1 \
  --distill_teacher_json "$KWS_PATH/distill_att_mh_rnn.json" \
  kws_transformer \
  --num_layers 12 \
  --heads 1 \
  --d_model 64 \
  --mlp_dim 256 \
  --dropout1 0. \
  --attention_type "time"

echo "Distillation training completed at $(date)"
