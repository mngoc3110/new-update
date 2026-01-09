#!/bin/bash
# Updated for Kaggle Environment
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
 echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# --- PATH CONFIGURATION ---
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"

# Experiment Name: Hierarchical (Binary Head) + EAA + IEC
EXP="Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-Resumed-STAGE3_EXTENDED" # Updated experiment name
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Hierarchical Training with EAA and IEC (100 Epochs)" # Updated echo

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu 0 \
  --seed 42 \
  --workers 4 \
  --print-freq 50 \
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
  --class-token-position "end" \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  \
  --use-hierarchical-prompt True \
  --use-adapter False \
  --use-iec False \
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
  --lambda-mi 0.5 --mi-warmup 5 --mi-ramp 0 \
  --lambda-dc 0.5 --dc-warmup 5 --dc-ramp 0 \
  --lambda-cons 0.1 \
  \
  --use-lsr2-loss True \
  --semantic-smoothing True \
  --smoothing-temp 0.1 \
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
  --stage3-epochs 90 \
  --stage3-logit-adjust-tau 0.8 \
  --stage3-logit-adjust-tau-neutral 0.1 \
  --stage3-max-class-weight 5.0 \
  --stage3-smoothing-temp 0.18 \
  \
  --stage4-logit-adjust-tau 0.1 \
  --stage4-max-class-weight 2.0 \
  --stage4-use-focal-loss False \
  --stage4-semantic-smoothing False \
  --inference-neutral-bias 0.0

# Note: Batch size 16 is a good starting point for Kaggle GPUs.
