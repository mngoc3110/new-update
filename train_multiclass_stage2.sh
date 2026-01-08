#!/bin/bash
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
 echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# Experiment Name
EXP="Multiclass_Stage2_Training"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

 echo "Starting STAGE 2: Multi-class Classification (4 Emotional Classes)"

# --- PATH CONFIGURATION ---
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
ANNOT_DIR="${ROOT_DIR}/RAER/annotation"
TRAIN_TXT="${ANNOT_DIR}/train_80.txt"
VAL_TXT="${ANNOT_DIR}/val_20.txt"
TEST_TXT="${ANNOT_DIR}/test.txt"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box" 
FACE_BOX="${BOX_DIR}/face.json"
BODY_BOX="${BOX_DIR}/body.json"
CLIP_PATH="ViT-B/32"
STAGE1_CHECKPOINT="outputs/Binary_Stage1_Training-[...]/model_best.pth" # <<< IMPORTANT: UPDATE THIS PATH

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
  --use-iec True \
  --emotional-only True \
  --load-stage1-checkpoint "${STAGE1_CHECKPOINT}" \
  \
  --epochs 30 \
  --batch-size 16 \
  \
  --lr 1e-3 \
  --lr-image-encoder 0 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 0 \
  --milestones 20 25 \
  --gamma 0.1 \
  \
  --lambda-binary 0.0 \
  --lambda-mi 0.5 --mi-warmup 5 \
  --lambda-dc 0.5 --dc-warmup 5 \
  --lambda-cons 0.1 \
  \
  --use-lsr2-loss True \
  --semantic-smoothing True \
  --use-focal-loss True \
  --focal-gamma 2.0 \
  --unfreeze-visual-last-layer False

# Note: In this stage, we freeze the visual backbone and adapter, and only train the classifier.