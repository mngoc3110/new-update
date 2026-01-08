#!/bin/bash
set -e

# --- CẤU HÌNH ĐƯỜNG DẪN (Chỉnh lại cho đúng với Kaggle của bạn) ---
ROOT_DIR="/Users/macbook/Downloads/week15/new-update 2"
ANNOT_DIR="${ROOT_DIR}/RAER/annotation"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"

# --- CHỌN MODEL ĐỂ ĐÁNH GIÁ ---
# Thay đổi đường dẫn này tới file model.pth (Epoch 100) hoặc model_best.pth mà bạn muốn test
# Ví dụ: /kaggle/input/hierarchical-vitb32-0-68/model.pth
CHECKPOINT_PATH="outputs/Hierarchical_ViTB32_LiteHiCroPL_2Stage_BinaryFocus_30Epochs-[01-08]-[00:52]/model_best.pth"

# --- THRESHOLD CHO BINARY HEAD ---
# Ngưỡng xác suất để coi là "Non-Neutral".
# Nếu Prob(Non-Neutral) < THRESHOLD => Dự đoán là Neutral.
# Gợi ý:
# - 0.5: Cân bằng
# - 0.8: Rất khó tính, chỉ khi chắc chắn mới bảo là Emotion => Neutral sẽ tăng cao.
THRESHOLD=0.54

echo "========================================================"
echo "STARTING SMART EVALUATION (POST-PROCESSING)"
echo "Model: ${CHECKPOINT_PATH}"
echo "Binary Threshold: ${THRESHOLD}"
echo "========================================================"

python main.py \
  --mode eval \
  --exper-name "Eval_Smart_Thresh${THRESHOLD}" \
  --eval-checkpoint "${CHECKPOINT_PATH}" \
  --inference-threshold-binary ${THRESHOLD} \
  --gpu 0 \
  --seed 42 \
  --workers 4 \
  --print-freq 10 \
  \
  --root-dir "${ROOT_DIR}" \
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
  --use-hierarchical-prompt True \
  --batch-size 16

echo "========================================================"
echo "EVALUATION COMPLETE"
echo "Check the log above for Confusion Matrix and UAR/WAR."
