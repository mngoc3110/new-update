#!/bin/bash
set -e

# Experiment Name: ViTB32 + Focal + StrongHead + StrongerSemanticSmoothing
EXP="ViTB32_4Stage_FullOptimization_100Epochs"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Stable Training: ViT-B/32 + Correct Logit Adjustment + 4-Stage Full Optimization (100 Epochs)"

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
  \
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
  --stage2-logit-adjust-tau 0.4 \
  --stage2-max-class-weight 2.0 \
  --stage2-smoothing-temp 0.15 \
  --stage2-label-smoothing 0.1 \
  \
  --stage3-epochs 70 \
  --stage3-logit-adjust-tau 0.8 \
  --stage3-max-class-weight 2.5 \
  --stage3-smoothing-temp 0.18 \
  \
  --stage4-logit-adjust-tau 0.5 \
  --stage4-max-class-weight 1.5

# 4-Stage Full Optimization (100 Epochs):
# Stage 1 (0-5): Short Warmup -> Stabilize quickly.
# Stage 2 (6-30): DRW -> Introduce minority classes early.
# Stage 3 (31-70): Targeted Push -> Aggressively boost Confusion/Enjoyment.
# Stage 4 (71-100): Cooldown -> Refine and balance WAR/UAR with low LR.

