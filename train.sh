#!/bin/bash
set -e

# Experiment Name
EXP="TwoStage_50ep_ProperSplit"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Two-Stage Training (50 Epochs): Safe Base (Stage 1) -> Controlled Boosting (Stage 2)"

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
  --temporal-layers 1 \
  \
  --epochs 50 \
  --batch-size 8 \
  \
  --lr 1e-2 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 1e-3 \
  --milestones 20 40 \
  --gamma 0.1 \
  \
  --lambda-mi 0.5 --mi-warmup 5 \
  --lambda-dc 0.5 --dc-warmup 5 \
  \
  --semantic-smoothing True \
  \
  --stage1-epochs 15 \
  --stage1-label-smoothing 0.05 \
  --stage1-smoothing-temp 0.1 \
  \
  --stage2-logit-adjust-tau 0.4 \
  --stage2-max-class-weight 3.0 \
  --stage2-smoothing-temp 0.1 \
  --stage2-label-smoothing 0.2

# Note: Paths (root-dir, annotations) are set to defaults relative to ./ 
# Please ensure they match your actual data location.
