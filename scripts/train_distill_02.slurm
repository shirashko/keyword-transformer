#!/bin/bash
#SBATCH --job-name=KWT_Distill_shir
#SBATCH --output=logs/kwt_distill_%j.out
#SBATCH --error=logs/kwt_error_distill_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00

# --- Environment Setup ---
KWS_PATH="/home/yandex/APDL2526a/rashkovits/keyword-transformer"
cd $KWS_PATH

# Ensure project root is in PYTHONPATH
export PYTHONPATH=$KWS_PATH:$PYTHONPATH

# Initialize Conda
source /home/yandex/APDL2526a/rashkovits/anaconda3/etc/profile.d/conda.sh
conda activate kwt_env_310

# --- Environment & GPU Setup ---
PYTHON_EXEC="/home/ADV_2526a/rashkovits/envs/kwt_env_310/bin/python"

module load cuda/12.2 2>/dev/null
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH="/home/ADV_2526a/rashkovits/envs/kwt_env_310/lib:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Compatibility Fixes
export TF_USE_LEGACY_KERAS=1
# Paths
DATA_PATH="$KWS_PATH/data2/speech_commands_v0.02"
MODELS_PATH="$KWS_PATH/models_data_v2_12_labels"
TRAIN_DIR="$MODELS_PATH/kwt1_distill/"
mkdir -p logs

# --- Checkpoint Resume Logic ---
# Prioritizes last_weights.ckpt if training finished, otherwise best_weights.ckpt.
START_CHECKPOINT_PARAM=""
if [ -f "${TRAIN_DIR}last_weights.ckpt.index" ]; then
    echo "Full distillation training was completed. Resuming from last_weights.ckpt"
    START_CHECKPOINT_PARAM="--start_checkpoint ${TRAIN_DIR}last_weights.ckpt"
elif [ -f "${TRAIN_DIR}best_weights.ckpt.index" ]; then
    echo "Distillation interrupted. Resuming from best_weights.ckpt"
    START_CHECKPOINT_PARAM="--start_checkpoint ${TRAIN_DIR}best_weights.ckpt"
else
    echo "No distillation checkpoint found. Starting from scratch."
fi

echo "Starting KWT Distillation Training at: $(date)"
echo "Teacher Model configuration: $KWS_PATH/distill_att_mh_rnn.json"

# --- Execution ---
# NOTE: --start_step 48500 is hardcoded from a specific interrupted training session.
# If resuming YOUR OWN training, update this value to match your checkpoint's actual step,
# or remove it to auto-detect from the checkpoint. Do NOT reuse this value blindly.
$PYTHON_EXEC -m kws_streaming.train.model_train_eval \
  $START_CHECKPOINT_PARAM \
   --start_step 48500 \
  --data_url "" \
  --data_dir "$DATA_PATH/" \
  --train_dir "$TRAIN_DIR" \
  --mel_upper_edge_hertz 7600 \
  --optimizer "adamw" \
  --lr_schedule "cosine" \
  --how_many_training_steps "120000" \
  --eval_step_interval 500 \
  --warmup_epochs 0 \
  --l2_weight_decay 0.0 \
  --learning_rate "0.0005" \
  --batch_size 100 \
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

echo "Distillation training session ended at $(date)"