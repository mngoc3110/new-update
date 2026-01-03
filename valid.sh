#!/bin/bash
set -e

CKPT="$1"
SPLIT="${2:-test}"   # val hoặc test

EXP="eval_${SPLIT}"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

python main.py \
  --mode eval \
  --exper-name "${EXP}" \
  --output-path "${OUT}" \
  --gpu cpu \
  --workers 0 \
  --use-amp False \
  --eval-checkpoint "${CKPT}" \
  --eval-split "${SPLIT}" \
  --root-dir ./ \
  --train-annotation RAER/annotation/train_80.txt \
  --val-annotation RAER/annotation/val_20.txt \
  --test-annotation RAER/annotation/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  --text-type class_descriptor \
  --contexts-number 12 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --batch-size 2 \
  2>&1 | tee "${OUT}/log.txt"