#!/bin/bash
set -e

# Experiment Name: ViTB32 + Focal + StrongHead + StrongerSemanticSemanticSmoothing
EXP="ViTB32_LiteHiCroPL_4Stage_BalancedPush_100Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Stable Training: ViT-B/32 + Lite-HiCroPL + 4-Stage Balanced Push (100 Epochs)"

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu mps \
  --seed 42 \
  --workers 4 \
  --print-freq 10 \
  \
  --root-dir ./ \
  --train-annotation RAER/annotation/train_80.txt \
  --val-annotation RAER/annotation/val_20.txt \
  --test-annotation RAER/annotation/test.txt \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
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
  --lr-prompt-learner 1e-3 \
  --milestones 70 90 \
  --gamma 0.1 \
  \
  --lambda-mi 0.5 --mi-warmup 5 \
  --lambda-dc 0.5 --dc-warmup 5 \
  --lambda-cons 0.1 \
  \
  --use-lsr2-loss True \
  --semantic-smoothing True \
  --use-focal-loss True \
  --focal-gamma 2.0 \
  --unfreeze-visual-last-layer False \
  \
  --stage1-epochs 5 \
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
  --stage4-max-class-weight 1.2

# 4-Stage "Balanced Push" Strategy with Lite-HiCroPL:
# Stage 1 (0-5): Warmup.
# Stage 2 (6-30): Intro Minority (Weight 1.5, Tau 0.2).
# Stage 3 (31-70): Moderate Push (Weight 3.0, Tau 0.5).
# Stage 4 (71-100): Gentle Cooldown (Weight 1.2, Tau 0.1). 