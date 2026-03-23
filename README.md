# Keyword Transformer (KWT) - TAU Final Project

**Course:** Advanced Topics in Audio Processing using Deep Learning  
**Instructor:** Tal Rosenwein  
**Team Members:** Shir Rashkovits, Shoham Mazuz, Gal Getz, Omer Ventura

---

## 📝 Overview
This repository contains our implementation and reproduction of the **Keyword Transformer (KWT)**, based on the research by Axel Berg et al. 

Our project specifically focuses on the **KWT-1** architecture (the lightweight variant) for the **12-label keyword spotting task** using the Google Speech Commands V2 dataset. We explore the reproducibility and efficacy of the model by comparing two training strategies:
1. **Baseline:** Training KWT-1 from scratch.
2. **Distillation:** Leveraging a pre-trained **Att-MH-RNN** teacher (CNN-RNN architecture) to guide the student Transformer.

Our analysis focuses on the comparison between a model trained from scratch and one trained with distillation.

<p align="center">
  <img src="assets/kwt_pipeline.png" width="700">
  <br>
  <em>Figure 1: KWT Architecture - From Mel-Spectrogram patches to Transformer Encoder outputs.</em>
</p>



---

## 📂 Repository Structure

```text
.
├── models_data_v2_12_labels/ # Training artifacts: Checkpoints (.ckpt), TensorBoard logs, and flags.
├── scripts/                  # Slurm/Bash scripts for remote training.
├── kws_streaming/            # Core logic: Model definitions (KWT, Att-MH-RNN), data loaders, and training loops.
├── assets/                   # Architecture diagrams.
├── graphs/                   # Training curves and result plots.
├── reports/                  # Training run analysis and results.
├── distill_att_mh_rnn.json   # Teacher model config for distillation.
├── eval_checkpoint.py        # Evaluation script for test set accuracy.
├── requirements.txt          # Python dependencies.
└── README.md                 # Project documentation.
```

---


## 🛠 Setup & Requirements
This project is built and tested using **Python 3.10** on **macOS (Apple Silicon)**.

1. **Environment Setup:**
```bash
python3.10 -m venv venv3
source venv3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

2. **Data Preparation:**
To download and extract the Google Speech Commands V2 dataset:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data2
mv ./speech_commands_v0.02.tar.gz ./data2
cd ./data2
tar -xf ./speech_commands_v0.02.tar.gz
cd ../

```

---

## 🚀 How to Run

### 1. Evaluation

To evaluate test set accuracy on saved checkpoints:

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

### 2. Training

**On the SLURM cluster:**
* **From Scratch:** `sbatch scripts/train_baseline.slurm`
* **With Distillation:** `sbatch scripts/train_distill.slurm` (Requires teacher weights in `models_data_v2_12_labels/att_mh_rnn_1/`)

**Locally:**
```bash
source venv3/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export TF_USE_LEGACY_KERAS=1

# Baseline KWT-1
python -m kws_streaming.train.model_train_eval \
  --data_url "" \
  --data_dir ./data2/speech_commands_v0.02/ \
  --train_dir ./models_data_v2_12_labels/kwt1_baseline/ \
  --train 1 --batch_size 256 --optimizer adamw --lr_schedule cosine \
  --learning_rate 0.0005 --how_many_training_steps 46876 \
  --mel_upper_edge_hertz 7600 --window_size_ms 30.0 --window_stride_ms 10.0 \
  --mel_num_bins 80 --dct_num_features 40 \
  kws_transformer --num_layers 12 --heads 1 --d_model 64 --mlp_dim 256 \
  --dropout1 0. --attention_type "time"

# Distillation
python -m kws_streaming.train.model_train_eval \
  --data_url "" \
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

---

## 📊 Results & Visualization

### **TensorBoard Logs**
To visualize training progress, loss curves, and accuracy from the initial 24-hour run, execute the following command from the project root:

```bash
tensorboard --logdir ./models_data_v2_12_labels/first_run_on_server/
```

> **Note:** This directory contains synchronous logs for both the **Baseline** and **Distillation** experiments, allowing for side-by-side comparison in the TensorBoard dashboard.

### **Checkpoints & Models**

The trained weights (checkpoints) and configuration flags for these experiments are organized as follows:

| Experiment | Directory Path |
| --- | --- |
| **Baseline (KWT1)** | `models_data_v2_12_labels/first_run_on_server/kwt1_baseline/` |
| **Distillation (KWT1)** | `models_data_v2_12_labels/first_run_on_server/kwt1_distill/` |

---

## ⚠️ Challenges & Modifications

* **Hardware Adaptation:** Optimized for **Apple Silicon** by adjusting batch sizes and memory usage.
* **Architecture Scaling:** Explicitly modified the pipeline to target the **KWT-1** variant ($d_{model}=64$, $h=1$).
* **Data Generator:** Patched the `unknown` class sampling logic to ensure compatibility with modern TensorFlow versions.
