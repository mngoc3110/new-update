#!/bin/bash
# Updated for Kaggle Environment
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# --- PATH CONFIGURATION ---
# Kaggle Paths (Default)
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"

# Experiment Name: Hierarchical (Binary Head) + LiteHiCroPL + 4-Stage
# Original experiment name
ORIGINAL_EXP="Hierarchical_ViTB32_LiteHiCroPL_4Stage_BalancedPush_100Epochs"

# Output directory for the RESUMED run
# This will create a new directory to store logs and checkpoints of the resumed training
RESUMED_OUT="outputs/${ORIGINAL_EXP}-Resumed-$(date +%m-%d-%H%M)"
mkdir -p "${RESUMED_OUT}"

# Path to the specific checkpoint to resume from.
# Ensure this path is correct for your setup.
# The user specified the model was saved at 'outputs/Hierarchical_ViTB32_LiteHiCroPL_4Stage_BalancedPush_100Epochs-[01-06]-[18:25]'
# And we are resuming from epoch 68.
CHECKPOINT_TO_RESUME="outputs/Hierarchical_ViTB32_LiteHiCroPL_4Stage_BalancedPush_100Epochs-[01-06]-[18:25]/checkpoint_68.pth"

echo "Resuming Hierarchical Training from: ${CHECKPOINT_TO_RESUME}"
echo "Outputs will be saved to: ${RESUMED_OUT}"

python main.py \
  --mode train \
  --exper-name "${ORIGINAL_EXP}" \
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
  \
  --stage4-logit-adjust-tau 0.1 \
  --stage4-max-class-weight 1.2 \
  \
  --resume "${CHECKPOINT_TO_RESUME}" \
  --output_dir "${RESUMED_OUT}" \
  --initial-stage 3
