# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案簡介
這是一個 YOLOv7-tiny 模型的 Baseline 實作專案，專注於在 COCO2017 資料集上進行 320×320 輸入尺寸的訓練與量化。

## YOLOv7 官方倉庫
- **GitHub**: https://github.com/WongKinYiu/yolov7
- **本地路徑**: `./yolov7/`
- **模型配置**: `./yolov7/cfg/training/yolov7-tiny.yaml`
- **預訓練權重**: 從官方 releases 或 README 提供的連結下載 `yolov7-tiny.pt`

## 關鍵技術規格
- **模型**: YOLOv7-tiny
- **資料集**: COCO 2017 (官方 split)，輸入統一 320×320 (letterbox)
- **訓練**: AMP (FP16 混合精度)，300 epochs
- **量化**: PTQ (ONNX Runtime 靜態量化，QDQ 格式)
- **評測標準**: COCO mAP50-95 (含 S/M/L)

## 開發環境要求
- **CUDA**: 12.1
- **PyTorch**: 2.2.2
- **torchvision**: 0.17.2
- **onnx**: 1.15.0
- **onnxruntime**: 1.17.1
- **onnxruntime-tools**: 1.7.0
- **onnxsim**: 0.4.36

## 核心工作流程

### 1. 資料準備
```bash
# 生成訓練/驗證/校正集清單
find /data/coco/train2017 -type f -name "*.jpg" | sort > train.txt
find /data/coco/val2017 -type f -name "*.jpg" | sort > val.txt
awk 'NR%10==1' val.txt | head -n 512 > calib.txt

# 生成 SHA256 校驗檔
xargs -a train.txt -I{} sha256sum "{}" > manifest_train.sha256
xargs -a val.txt -I{} sha256sum "{}" > manifest_val.sha256
xargs -a calib.txt -I{} sha256sum "{}" > manifest_calib.sha256
```

### 2. 模型訓練
```bash
# 使用 YOLOv7 官方訓練腳本
cd yolov7
python train.py --img 320 --batch 128 --epochs 300 \
  --cfg cfg/training/yolov7-tiny.yaml \
  --data data/coco.yaml --weights yolov7-tiny.pt \
  --device 0 --workers 4 --hyp data/hyp.scratch.tiny.yaml

# 背景驗證監控 (每 60 秒檢查)
while true; do
  L=./runs/train/exp/weights/last.pt
  [ -f "$L" ] && python test.py --weights $L --data data/coco.yaml \
    --img 320 --conf-thres 0.001 --iou-thres 0.65 \
    --batch 1 --device 0
  sleep 60
done &
```

### 3. ONNX 導出
```bash
python export.py --weights runs/train/exp/weights/best.pt \
  --img 320 320 --batch 1 --include onnx --opset 13
python -m onnxsim model.onnx model-sim.onnx
md5sum model.onnx model-sim.onnx > onnx_md5.txt
```

### 4. INT8 量化 (PTQ)
```bash
python tools/ort_ptq.py --model model.onnx --calib calib.txt \
  --out model-int8.onnx --method percentile
md5sum model-int8.onnx > onnx_int8_md5.txt
```

### 5. 模型評測
```bash
python tools/eval_onnx.py --model model.onnx --img 320 \
  --report eval_report.json
python tools/eval_onnx.py --model model-int8.onnx --img 320 \
  --report eval_report.json --append
```

## 專案架構
- **yolov7/**: YOLOv7 官方程式碼庫
  - `train.py`: 訓練主程式
  - `test.py`: 驗證/測試程式
  - `detect.py`: 推論程式
  - `export.py`: 模型導出程式
  - `cfg/training/`: 模型配置檔案
  - `data/`: 資料集配置和超參數檔案
  - `models/`: 模型定義程式碼
  - `utils/`: 工具函數
- **tools/**: 自定義工具腳本
  - `gen_lists.py`: 生成訓練/驗證/校正集清單
  - `ort_ptq.py`: ONNX Runtime PTQ 量化腳本
  - `eval_onnx.py`: ONNX 模型評測腳本
  - `val_watcher.sh`: 背景驗證監控腳本
- **runs/**: 訓練輸出目錄

## 重要參數設定

### 訓練參數 (基於 hyp.scratch.tiny.yaml)
- **Optimizer**: SGD
  - lr0: 0.01 (初始學習率)
  - lrf: 0.01 (最終學習率因子)
  - momentum: 0.937
  - weight_decay: 0.0005
- **Warmup**:
  - warmup_epochs: 3.0
  - warmup_momentum: 0.8
  - warmup_bias_lr: 0.1
- **Loss weights**:
  - box: 0.05
  - cls: 0.5
  - obj: 1.0
  - loss_ota: 1 (使用 ComputeLossOTA)
- **資料增強**:
  - HSV: h=0.015, s=0.7, v=0.4
  - Translate: 0.1
  - Scale: 0.5
  - Fliplr: 0.5
  - Mosaic: 1.0
  - Mixup: 0.05
  - 注意: Epoch 261-300 需手動關閉 Mosaic/MixUp (收尾階段)

### 前處理設定
- letterbox: auto=False, scaleFill=False, scaleup=True
- color: (114,114,114)
- stride: 32
- normalize: 1/255

### NMS 參數
- conf_thres: 0.001
- iou_thres: 0.65
- max_det: 300
- class_agnostic: False

### 量化設定
- **格式**: QDQ
- **Weights**: INT8, symmetric, per-channel
- **Activations**: INT8, asymmetric, per-tensor
- **Calibration**: Percentile 99.99% 或 MinMax
- **校正集**: 512 張 (從 val 集抽樣，無增強)

## 評測標準與預期結果
- **mAP50-95 (AMP/FP16)**: ~33-35
- **INT8 全量化掉點**: -1.5 ~ -3.0 mAP
- **INT8 混精度掉點**: -0 ~ -1.5 mAP (首層 + Detect head 回 FP16)
- **延遲**: INT8 約 FP16 的 0.5-0.7× (依硬體而異)
- **跨平台一致性**: 各引擎對 ORT-CPU 的差 ≤ 0.5 mAP

## 交付檔案清單
- `best.pt`: 訓練最佳權重 (AMP)
- `model.onnx`, `model-sim.onnx`: 原始與簡化 ONNX
- `model-int8.onnx`: INT8 量化模型
- `onnx_md5.txt`, `onnx_int8_md5.txt`: MD5 校驗檔
- `train.txt`, `val.txt`, `calib.txt`: 資料清單
- `manifest_*.sha256`: SHA256 校驗檔
- `eval_report.json`: 評測報告
- `ENV.md`: 環境版本記錄

## 開發原則
- **漸進式開發**: 每個模組完成後必須驗證才進入下一個
- **STOP 規則**: 完成每個函數後暫停驗證，避免一次寫太多程式碼
- **真實資料測試**: 優先使用真實資料而非假資料進行測試
- **透明溝通**: 所有設計決策都要說明理由
- **遵循 Baseline Spec**: 嚴禁改動資料分割、前處理、訓練 epoch、AMP 開關、PTQ 流程、校正集、NMS 參數

## 允許的改動範圍
- 模型架構優化
- Loss 函數改進
- 資料增強策略 (但收尾關閉 Mosaic/MixUp 的節點不變)

全部用繁體中文回答