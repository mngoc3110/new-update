#!/bin/bash
set -e

# Usage: ./valid.sh <CHECKPOINT_PATH_OR_DIR> [SPLIT] [CLIP_PATH] [INFERENCE_THRESHOLD]
# Example: ./valid.sh outputs/Experiment_Name  (defaults to val, ViT-B/32, -1.0)
# Example: ./valid.sh outputs/Experiment_Name/model_best.pth test ViT-B/16 0.7

INPUT_PATH="$1"
SPLIT="${2:-val}"   # val hoặc test
CLIP_MODEL="${3:-ViT-B/32}" # Mặc định là ViT-B/32, có thể truyền ViT-B/16
INFERENCE_THRESHOLD="${4:--1.0}" # Mặc định là -1.0 (tắt tính năng này), truyền số > 0 để bật (ví dụ 0.5)

if [ -z "$INPUT_PATH" ]; then
  echo "Error: Please provide a checkpoint path or experiment directory."
  echo "Usage: $0 <path_to_checkpoint_or_dir> [split: val|test] [clip_model] [inference_threshold]"
  exit 1
fi

# Xử lý đường dẫn: Nếu là thư mục, tự động tìm model_best.pth
if [ -d "$INPUT_PATH" ]; then
  CKPT="${INPUT_PATH}/model_best.pth"
  if [ ! -f "$CKPT" ]; then
    echo "Warning: model_best.pth not found in directory. Trying model.pth..."
    CKPT="${INPUT_PATH}/model.pth"
  fi
else
  CKPT="$INPUT_PATH"
fi

if [ ! -f "$CKPT" ]; then
  echo "Error: Checkpoint file not found: $CKPT"
  exit 1
fi

echo "=> Evaluating Checkpoint: $CKPT"
echo "=> Split: $SPLIT"
echo "=> CLIP Model: $CLIP_MODEL"
echo "=> Inference Threshold: $INFERENCE_THRESHOLD"

# Xác định file annotation dựa trên split
if [ "$SPLIT" == "val" ]; then
  TEST_ANNOT="RAER/annotation/val_20.txt"
else
  TEST_ANNOT="RAER/annotation/test.txt"
fi

echo "=> Target Annotation: $TEST_ANNOT"

# Tạo thư mục output
# Lưu ý: main.py tự tạo output path dựa trên exper-name trong setup_paths_and_logging,
# nhưng chúng ta vẫn cần mkdir để tee log.
# Để tránh xung đột, ta đặt exper-name sao cho main.py tạo ra đúng thư mục ta muốn hoặc chấp nhận thư mục do main.py tạo.
# Tuy nhiên, để đơn giản và tránh lỗi "unrecognized arguments", ta sẽ để main.py tự xử lý output path
# và chỉ lấy log ra stdout/tee.

# Sửa lại tên experiment để main.py tạo thư mục output hợp lý
EXPER_NAME="eval_${SPLIT}_thresh${INFERENCE_THRESHOLD}_$(date +%H%M)"

echo "=> CKPT Directory: $CKPT_DIR"
echo "=> Experiment Name: $EXPER_NAME"

# Chạy python trực tiếp, không qua tee để tránh lỗi file system và xem output ngay
python main.py \
  --mode eval \
  --exper-name "${EXPER_NAME}" \
  --gpu cpu \
  --workers 0 \
  --use-amp False \
  --eval-checkpoint "${CKPT}" \
  --root-dir ./ \
  --train-annotation RAER/annotation/train_80.txt \
  --val-annotation RAER/annotation/val_20.txt \
  --test-annotation "${TEST_ANNOT}" \
  --clip-path "${CLIP_MODEL}" \
  --inference-threshold "${INFERENCE_THRESHOLD}" \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  --text-type class_descriptor \
  --contexts-number 16 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --use-hierarchical-prompt True \
  --temporal-layers 2 \
  --batch-size 8

echo "=> Done."
