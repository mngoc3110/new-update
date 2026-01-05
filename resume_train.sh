#!/bin/bash
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# --- RESUME CONFIGURATION ---
# Experiment Name (Appended "-Resumed" to distinguish)
EXP="Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-Resumed-STAGE3_EXTENDED"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

# --- CHECKPOINT PATH ---
# IMPORTANT: Update this path to where your model.pth is located on Kaggle.
# Example: If you upload the 'outputs' folder as a dataset, it might be:
# /kaggle/input/my-prev-output/outputs/Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-[01-05]-[01:04]/model.pth
# Below is the local path relative to your current folder structure for reference.
CHECKPOINT_PATH="outputs/Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-[01-05]-[01:04]/model.pth"

echo "Resuming Training from: ${CHECKPOINT_PATH}"

# --- PATH CONFIGURATION (Same as Kaggle Train) ---
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
  --resume "${CHECKPOINT_PATH}" \
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
  \
  --epochs 100 \
  --batch-size 16 \
  \
  --lr 1e-3 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 5e-4 \
  --milestones 70 90 \
  --gamma 0.1 \
  \
  --lambda-mi 0.5 --mi-warmup 5 \
  --lambda-dc 0.5 --dc-warmup 5 \
  --lambda-cons 0.1 \
  \
  --use-lsr2-loss True \
  \
  --semantic-smoothing True \
  --use-focal-loss True \
  --focal-gamma 2.0 \
  --unfreeze-visual-last-layer False \
  \
  --stage1-epochs 3 \
  --stage1-label-smoothing 0.05 \
  --stage1-smoothing-temp 0.15 \
  \
  --stage2-epochs 30 \
  --stage2-logit-adjust-tau 0.5 \
  --stage2-max-class-weight 2.0 \
  --stage2-smoothing-temp 0.15 \
  --stage2-label-smoothing 0.1 \
  \
  --stage3-epochs 70 \
  --stage3-logit-adjust-tau 0.8 \
  --stage3-max-class-weight 5.0 \
  --stage3-smoothing-temp 0.18 \
  \
  --stage4-logit-adjust-tau 0.8 \
  --stage4-max-class-weight 5.0