#!/bin/bash
# Updated for Kaggle Environment
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
 echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# --- PATH CONFIGURATION ---
ROOT_DIR="/content/drive/MyDrive/khoaluan/Dataset/"
ANNOT_DIR="${ROOT_DIR}/RAER/annotation"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"

# Experiment Name: Hierarchical (Binary Head) + EAA + IEC
EXP="Hierarchical_ViTB32_EAA_IEC_30Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Hierarchical Training with EAA and IEC"

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu mps \
  --seed 42 \
  --workers 4 \
  --print-freq 10 \
  \
  --root-dir "${ROOT_DIR}" \
  --train-annotation "${ANNOT_DIR}/train_80.txt" \
  --val-annotation "${ANNOT_DIR}/val_20.txt" \
  --test-annotation "${ANNOT_DIR}/test.txt" \
  --bounding-box-face "${BOX_DIR}/face.json" \
  --bounding-box-body "${BOX_DIR}/body.json" \
  \
  --clip-path ViT-B/32 \
  --text-type class_descriptor \
  --image-size 224 \
  --num-segments 16 \
  --duration 1 \
  --temporal-layers 2 \
  --contexts-number 16 \
  \
  --use-hierarchical-prompt True \
  --use-adapter True \
  --use-iec True \
  \
  --epochs 30 \
  --batch-size 16 \
  \
  --lr 1e-3 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-3 \
  --milestones 20 25 \
  --gamma 0.1 \
  \
  --lambda-mi 0.5 --mi-warmup 5 \
  --lambda-dc 0.5 --dc-warmup 5 \
  --lambda-cons 0.1 \
  --lambda-binary 2.0 \
  --distraction-boost 1.5 \
  \
  --use-lsr2-loss True \
  --semantic-smoothing True \
  --use-focal-loss True \
  --focal-gamma 2.0 \
  --unfreeze-visual-last-layer False \
  \
  --stage1-epochs 5 \
  --stage1-label-smoothing 0.1 \
  --stage1-smoothing-temp 0.15 \
  \
  --stage2-epochs 25 \
  --stage2-logit-adjust-tau 0.2 \
  --stage2-max-class-weight 1.5 \
  --stage2-smoothing-temp 0.15 \
  --stage2-label-smoothing 0.1

# Note: Batch size 8 for MPS stability. Increase to 16/32 if on strong GPU.