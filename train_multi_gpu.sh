#!/bin/bash
# 多 GPU 訓練腳本（支援 DDP）

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
YOLOV7_DIR="$PROJECT_ROOT/yolov7"

# 檢測 GPU 數量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "檢測到 $NUM_GPUS 個 GPU"

# 根據 GPU 數量調整參數
if [ $NUM_GPUS -eq 1 ]; then
    BATCH_SIZE=128
    LR=0.01
elif [ $NUM_GPUS -eq 2 ]; then
    BATCH_SIZE=256
    LR=0.02
elif [ $NUM_GPUS -eq 4 ]; then
    BATCH_SIZE=512
    LR=0.04
elif [ $NUM_GPUS -eq 8 ]; then
    BATCH_SIZE=1024
    LR=0.08
else
    BATCH_SIZE=$((128 * NUM_GPUS))
    LR=$(echo "scale=3; 0.01 * $NUM_GPUS" | bc)
fi

echo "使用配置："
echo "  Batch Size: $BATCH_SIZE (每個 GPU: $((BATCH_SIZE/NUM_GPUS)))"
echo "  Learning Rate: $LR"

cd "$YOLOV7_DIR"

# 單 GPU
if [ $NUM_GPUS -eq 1 ]; then
    python train.py \
        --img 320 \
        --batch $BATCH_SIZE \
        --epochs 300 \
        --cfg cfg/training/yolov7-tiny.yaml \
        --data data/coco.yaml \
        --weights yolov7-tiny.pt \
        --device 0 \
        --hyp data/hyp.scratch.tiny.yaml \
        --name "yolov7-tiny-320-bs${BATCH_SIZE}"
else
    # 多 GPU (DDP)
    python -m torch.distributed.launch \
        --nproc_per_node $NUM_GPUS \
        --master_port 9527 \
        train.py \
        --img 320 \
        --batch $BATCH_SIZE \
        --epochs 300 \
        --cfg cfg/training/yolov7-tiny.yaml \
        --data data/coco.yaml \
        --weights yolov7-tiny.pt \
        --device 0,1,2,3,4,5,6,7 \
        --sync-bn \
        --hyp data/hyp.scratch.tiny.yaml \
        --name "yolov7-tiny-320-${NUM_GPUS}gpu-bs${BATCH_SIZE}"
fi