#!/bin/bash
# YOLOv7-tiny 測試腳本

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
YOLOV7_DIR="$PROJECT_ROOT/yolov7"

cd "$YOLOV7_DIR"

echo "工作目錄：$(pwd)"
echo "開始測試..."

python test.py \
    --img 320 \
    --batch 1 \
    --conf-thres 0.001 \
    --iou-thres 0.65 \
    --device 0 \
    --data data/coco.yaml \
    --weights "${1:-runs/train/yolov7-tiny-320/weights/best.pt}" \
    "$@"