#!/bin/bash
#SBATCH --job-name=KWT_Baseline_shir
#SBATCH --output=logs/kwt_train_%j.out
#SBATCH --error=logs/kwt_error_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# --- Environment Setup ---
KWS_PATH="/home/yandex/APDL2526a/rashkovits/keyword-transformer"
cd $KWS_PATH

# Ensure project root is in PYTHONPATH for module discovery
export PYTHONPATH=$KWS_PATH:$PYTHONPATH

# Initialize Conda
source /home/yandex/APDL2526a/rashkovits/anaconda3/etc/profile.d/conda.sh
conda activate kwt_env_310

module load cuda/12.2 2>/dev/null || echo "CUDA module not found, relying on manual paths"

# Compatibility Fixes
pip install --quiet tf-keras==2.15.0
export TF_USE_LEGACY_KERAS=1

# CUDA Environment
export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

nvidia-smi

# Paths
DATA_PATH="$KWS_PATH/data2/speech_commands_v0.02"
MODELS_PATH="$KWS_PATH/models_data_v2_12_labels"
mkdir -p logs

# Clean old run to avoid FileExistsError
rm -rf "$MODELS_PATH/kwt1_baseline/"

echo "Starting KWT Baseline Training..."
echo "Start time: $(date)"

# --- Execution ---
python -m kws_streaming.train.model_train_eval \
  --data_url "" \
  --data_dir "$DATA_PATH/" \
  --train_dir "$MODELS_PATH/kwt1_baseline/" \
  --mel_upper_edge_hertz 7600 \
  --optimizer "adamw" \
  --lr_schedule "cosine" \
  --how_many_training_steps "46876" \
  --eval_step_interval 144 \
  --warmup_epochs 10 \
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

echo "Training completed at $(date)"