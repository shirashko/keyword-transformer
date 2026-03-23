#!/bin/bash
#SBATCH --job-name=KWT_Baseline_shir
#SBATCH --output=logs/kwt_train_%j.out
#SBATCH --error=logs/kwt_error_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00

# --- Environment Setup ---
KWS_PATH="/home/yandex/APDL2526a/rashkovits/keyword-transformer"
cd $KWS_PATH
export PYTHONPATH=$KWS_PATH:$PYTHONPATH

# Activate Conda
source /home/yandex/APDL2526a/rashkovits/anaconda3/etc/profile.d/conda.sh
conda activate kwt_env_310

# Paths
DATA_PATH="$KWS_PATH/data2/speech_commands_v0.02"
MODELS_PATH="$KWS_PATH/models_data_v2_12_labels"
TRAIN_DIR="$MODELS_PATH/kwt1_baseline/"
mkdir -p logs

# --- GPU & CUDA Setup ---
module load cuda/12.2 2>/dev/null
module load cudnn 2>/dev/null

export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true

# --- Checkpoint Logic ---
START_CHECKPOINT_PARAM=""
if [ -f "${TRAIN_DIR}last_weights.ckpt.index" ]; then
    echo "Full training was completed. Resuming from last_weights.ckpt"
    START_CHECKPOINT_PARAM="--start_checkpoint ${TRAIN_DIR}last_weights.ckpt"
elif [ -f "${TRAIN_DIR}best_weights.ckpt.index" ]; then
    echo "Training was interrupted. Resuming from best_weights.ckpt"
    START_CHECKPOINT_PARAM="--start_checkpoint ${TRAIN_DIR}best_weights.ckpt"
else
    echo "No checkpoint found. Starting training from scratch."
fi

echo "Starting KWT Training at: $(date)"

# --- Execution ---
python -m kws_streaming.train.model_train_eval \
  $START_CHECKPOINT_PARAM \
  --start_step 43056 \
  --data_url "" \
  --data_dir "$DATA_PATH/" \
  --train_dir "$TRAIN_DIR" \
  --mel_upper_edge_hertz 7600 \
  --optimizer "adamw" \
  --lr_schedule "cosine" \
  --how_many_training_steps "46876" \
  --eval_step_interval 144 \
  --warmup_epochs 0 \
  --l2_weight_decay 0.1 \
  --learning_rate "0.0005" \
  --batch_size 256 \
  --label_smoothing 0.1 \
  --window_size_ms 30.0 \
  --window_stride_ms 10.0 \
  --mel_num_bins 80 \
  --dct_num_features 40 \
  --alsologtostderr \
  --resample 0.15 \
  --train 1 \
  --use_spec_augment 1 \
  --time_masks_number 2 \
  --time_mask_max_size 25 \
  --frequency_masks_number 2 \
  --frequency_mask_max_size 7 \
  --pick_deterministically 1 \
  kws_transformer \
  --num_layers 12 \
  --heads 1 \
  --d_model 64 \
  --mlp_dim 256 \
  --dropout1 0. \
  --attention_type "time"

echo "Training session ended at $(date)"