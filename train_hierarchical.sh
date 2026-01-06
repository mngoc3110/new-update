#!/bin/bash
set -e

# --- PATH CONFIGURATION ---
# Default to local development paths
ROOT_DIR="./"
ANNOT_DIR="RAER/annotation"
BOX_DIR="RAER/bounding_box"

# Uncomment these lines if running on Kaggle and datasets are mounted as specified
# ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
# ANNOT_DIR="/kaggle/input/raer-annot/annotation" # Assuming annotations are in a separate dataset
# BOX_DIR="${ROOT_DIR}/RAER/bounding_box"

# Experiment Name: Hierarchical (Binary Head) + LiteHiCroPL + 4-Stage
EXP="Hierarchical_ViTB32_LiteHiCroPL_4Stage_BalancedPush_100Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Hierarchical Training: Neutral vs Non-Neutral + 5-Class Classification"

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu 0 \
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
  \
  --epochs 100 \
  --batch-size 8 \
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
  --lambda-binary 1.0 \
  --distraction-boost 1.5 \
  \
  --use-lsr2-loss True \
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
  --stage2-logit-adjust-tau 0.2 \
  --stage2-max-class-weight 1.5 \
  --stage2-smoothing-temp 0.15 \
  --stage2-label-smoothing 0.1 \
  \
  --stage3-epochs 70 \
  --stage3-logit-adjust-tau 0.5 \
  --stage3-max-class-weight 3.0 \
  --stage3-smoothing-temp 0.18 \
  \  --stage4-logit-adjust-tau 0.1 \
  --stage4-max-class-weight 1.2

# Note: Batch size 8 for MPS stability. Increase to 16/32 if on strong GPU.
