#!/bin/bash
set -e

# --- SETUP ENVIRONMENT ---
echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# Experiment Name
EXP="MultiTask_ViTB32_Balanced_40Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Training with Multi-Task (5-class + binary) Strategy"

# --- PATH CONFIGURATION ---
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset" 
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
TRAIN_TXT="${ANNOT_DIR}/train_80.txt"
VAL_TXT="${ANNOT_DIR}/val_20.txt"
TEST_TXT="${ANNOT_DIR}/test.txt"
BOX_DIR="/kaggle/input/raer-video-emotion-dataset/RAER/bounding_box" 
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
  \
  --epochs 40 \
  --batch-size 16 \
  \
  --lr 5e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --milestones 25 35 \
  --gamma 0.1 \
  \
  --lambda-binary 0.3 \
  --lambda-mi 0.0 \
  --lambda-dc 0.0 \
  --lambda-cons 0.0 \
  --use-focal-loss True

# Note: This script runs the recommended stable multi-task strategy.
# It uses a weighted CE loss for the main 5-class task and an auxiliary
# binary cross-entropy loss to help regularize the Neutral class boundary.
