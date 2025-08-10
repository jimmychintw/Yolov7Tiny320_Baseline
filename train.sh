#!/bin/bash
# YOLOv7-tiny 訓練腳本
# 解決路徑問題的包裝腳本

set -e

# 專案根目錄
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
YOLOV7_DIR="$PROJECT_ROOT/yolov7"

# 檢查環境
if [ ! -d "$YOLOV7_DIR" ]; then
    echo "錯誤：找不到 yolov7 目錄"
    echo "請執行：git submodule update --init --recursive"
    exit 1
fi

# 進入 YOLOv7 目錄執行
cd "$YOLOV7_DIR"

echo "工作目錄：$(pwd)"
echo "開始訓練 YOLOv7-tiny..."

# 執行訓練
python train.py \
    --img 320 \
    --batch 128 \
    --epochs 300 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --data data/coco.yaml \
    --weights yolov7-tiny.pt \
    --device 0 \
    --workers 4 \
    --hyp data/hyp.scratch.tiny.yaml \
    --name yolov7-tiny-320 \
    "$@"  # 允許傳入額外參數

echo "訓練完成！"
echo "權重檔案位置：$YOLOV7_DIR/runs/train/yolov7-tiny-320/weights/"