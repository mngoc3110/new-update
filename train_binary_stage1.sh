#!/bin/bash
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---

echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# Experiment Name
EXP="Binary_Stage1_Training"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting STAGE 1: Binary Classification (Neutral vs. Non-Neutral)"

# --- PATH CONFIGURATION ---
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
  --use-iec False \
  --binary-classification-stage True \
  \
  --epochs 20 \
  --batch-size 16 \
  \
  --lr 1e-3 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-3 \
  --milestones 15 18 \
  --gamma 0.1 \
  \
  --lambda-binary 1.0

# Note: Only lambda_binary is used in this stage.
# Other losses are disabled within the loss function itself when binary_classification_stage is True.