#!/bin/bash
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# Experiment Name
EXP="Kaggle_ViTB32_EAA_IEC_30Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Kaggle Training with EAA and IEC"

# --- PATH CONFIGURATION ---
# Adjust these paths if your Kaggle dataset structure is different
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset" 
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
TRAIN_TXT="${ANNOT_DIR}/train_80.txt"
VAL_TXT="${ANNOT_DIR}/val_20.txt"
TEST_TXT="${ANNOT_DIR}/test.txt"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box" 
FACE_BOX="${BOX_DIR}/face.json"
BODY_BOX="${BOX_DIR}/body.json"
CLIP_PATH="ViT-B/32" 

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu 0 \
  --seed 42 \
  --workers 4 \
  --print-freq 50 \
  \
  --root-dir "${ROOT_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  \
  --clip-path "${CLIP_PATH}" \
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

# Note: Batch size 16 is a good starting point for Kaggle GPUs.
