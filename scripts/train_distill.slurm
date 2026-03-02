#!/bin/bash
#SBATCH --job-name=KWT_Distill_shir
#SBATCH --output=logs/kwt_distill_%j.out
#SBATCH --error=logs/kwt_error_distill_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# --- Environment Setup ---
KWS_PATH="/home/yandex/APDL2526a/rashkovits/keyword-transformer"
cd $KWS_PATH

export CUDA_VISIBLE_DEVICES=0
# Debug info
echo "Checking GPU status..."
nvidia-smi

# Ensure project root is in PYTHONPATH
export PYTHONPATH=$KWS_PATH:$PYTHONPATH

# Initialize Conda
source /home/yandex/APDL2526a/rashkovits/anaconda3/etc/profile.d/conda.sh
conda activate kwt_env_310

module load cuda/12.2 2>/dev/null || echo "CUDA module not found"

# Compatibility Fixes
export TF_USE_LEGACY_KERAS=1
export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Paths
DATA_PATH="$KWS_PATH/data2/speech_commands_v0.02"
MODELS_PATH="$KWS_PATH/models_data_v2_12_labels"
mkdir -p logs

# Clean old run directory
rm -rf "$MODELS_PATH/kwt1_distill/"

echo "Starting KWT Distillation Training (Teacher: Att-MH-RNN)..."
echo "Start time: $(date)"

# --- Execution ---
# Note: Using parameters optimized for distillation as per paper guidelines
python -m kws_streaming.train.model_train_eval \
  --data_url "" \
  --data_dir "$DATA_PATH/" \
  --train_dir "$MODELS_PATH/kwt1_distill/" \
  --mel_upper_edge_hertz 7600 \
  --optimizer "adamw" \
  --lr_schedule "cosine" \
  --how_many_training_steps "120000" \
  --eval_step_interval 500 \
  --warmup_epochs 5 \
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

echo "Distillation training completed at $(date)"