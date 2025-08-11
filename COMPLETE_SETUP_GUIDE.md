# YOLOv7-tiny Baseline 完整設定指南

這是一個完整的 YOLOv7-tiny 320×320 訓練環境設定指南，包含資料集準備、訓練和量化的全流程。

## 📋 目錄

- [系統需求](#系統需求)
- [快速開始](#快速開始)
- [詳細安裝步驟](#詳細安裝步驟)
- [資料集設定](#資料集設定)
- [訓練指令](#訓練指令)
- [參數調整](#參數調整)
- [常見問題](#常見問題)

## 🔧 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU（建議 RTX 4090 或以上）
- **VRAM**: 最少 16GB（推薦 24GB 以上）
- **RAM**: 32GB 以上
- **儲存空間**: 50GB 以上

### 軟體需求
- **作業系統**: Linux/macOS/Windows (WSL2)
- **Python**: 3.8-3.11
- **CUDA**: 12.1+
- **Git**: 2.0+

## 🚀 快速開始

```bash
# 1. Clone 專案
git clone https://github.com/your-repo/Yolov7Tiny320_Baseline.git
cd Yolov7Tiny320_Baseline

# 2. 設定環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安裝套件
pip install -r requirements.txt

# 4. 初始化子模組
git submodule update --init --recursive

# 5. 準備 COCO 資料集（解壓縮 coco.tar.gz 到 data/coco）
bash tools/unpack_coco.sh

# 6. 生成資料清單（預設使用專案內的 data/coco）
python tools/gen_lists.py  # 自動使用 data/coco 目錄

# 7. 開始訓練（會檢查資料集，不存在會報錯）
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml
```

## 📦 詳細安裝步驟

### 步驟 1：Clone 專案

```bash
# Clone 專案倉庫
git clone https://github.com/your-repo/Yolov7Tiny320_Baseline.git
cd Yolov7Tiny320_Baseline

# 檢查專案結構
ls -la
```

### 步驟 2：建立虛擬環境

```bash
# 建立 Python 虛擬環境
python -m venv venv

# 啟用虛擬環境
# Linux/macOS:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

# 確認環境
which python  # 應該指向 venv/bin/python
```

### 步驟 3：安裝依賴套件

```bash
# 升級 pip
pip install --upgrade pip

# 安裝專案依賴
pip install -r requirements.txt

# 驗證重要套件
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 步驟 4：初始化 YOLOv7 子模組

```bash
# 初始化並更新子模組
git submodule update --init --recursive

# 檢查 YOLOv7 目錄
ls yolov7/
# 應該看到：train.py, test.py, models/, utils/ 等

# 進入 yolov7 目錄下載預訓練權重
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
cd ..
```

## 💾 資料集設定

### 重要說明：資料集路徑與防止自動下載

本專案的資料集應放置在**專案根目錄**的 `data/coco` 目錄內：

```
Yolov7Tiny320_Baseline/
├── data/
│   └── coco/           # COCO 資料集位置（專案內）
│       ├── train2017/  # 訓練圖片
│       ├── val2017/    # 驗證圖片
│       └── annotations/ # 標註檔案
├── yolov7/            # YOLOv7 官方程式碼
└── tools/             # 工具腳本
```

**防止自動下載機制**：
- 訓練腳本會在開始前檢查資料集是否存在
- 如果資料集不存在，會直接報錯並退出，**不會自動下載**
- 這確保訓練環境的可控性和避免意外的網路下載

## 💾 資料集設定

### 重要：安全訓練機制

本專案實作了**安全訓練機制**，確保：

1. **資料集預檢查**：訓練前自動檢查資料集是否存在
2. **禁止自動下載**：資料集不存在時直接報錯，不會觸發 YOLOv7 的自動下載
3. **明確錯誤提示**：提供詳細的錯誤訊息和解決方案

使用安全訓練腳本：
```bash
# 使用 safe_train.py（完整檢查）
python tools/safe_train.py --epochs 300

# 使用 adaptive_train.py（包含檢查）
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml
```

### 方法 1：使用壓縮檔（推薦）

如果你有 `coco.tar.gz` 壓縮檔：

```bash
# 1. 將壓縮檔放入 data 目錄
mv ~/Downloads/coco.tar.gz data/

# 2. 解壓縮
bash tools/unpack_coco.sh

# 3. 驗證
ls data/coco/
# 應該看到：train2017/, val2017/, annotations/

# 4. 生成訓練清單（自動偵測 data/coco）
python tools/gen_lists.py  # 預設使用專案內的 data/coco

# 5. 確認清單檔案
wc -l data/*.txt
#   500 data/calib.txt
# 118287 data/train.txt
#  5000 data/val.txt
```

### 方法 2：手動下載 COCO 資料集

```bash
# 建立目錄
mkdir -p data/coco

# 下載 COCO 2017
cd data/coco

# 訓練圖片 (19GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 驗證圖片 (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 標註檔案 (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# 清理壓縮檔
rm *.zip

cd ../..

# 生成清單（預設使用專案內的 data/coco）
python tools/gen_lists.py  # 自動使用 data/coco

# 或指定其他路徑
python tools/gen_lists.py --coco-path /custom/path/coco
```

## 🔥 訓練指令

### 自動模式（推薦）

```bash
# 自動偵測 GPU 並選擇最佳參數
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml

# 執行過程會顯示：
# 檢查資料集配置: ../data/coco_local.yaml
#   ✓ train: ../data/coco/train2017 (118287 張圖片)
#   ✓ val: ../data/coco/val2017 (5000 張圖片)
# ✓ 資料集檢查通過
# 
# 檢測到 GPU: NVIDIA RTX 4090
# VRAM: 24.0 GB
# 自動選擇 batch size: 128
# 自動計算學習率: 0.0100

# 如果資料集不存在，會顯示：
# ❌ 錯誤：找不到以下資料集路徑：
#   - train: /path/to/missing/data
# 請先準備好資料集...
# (程式直接退出，不會開始訓練)
```

### 手動指定參數

```bash
# 指定 batch size 和學習率
python tools/adaptive_train.py \
    --batch-size 256 \
    --lr 0.02 \
    --epochs 300 \
    --img 320 \
    --data ../data/coco_local.yaml

# 多 GPU 訓練
./train_multi_gpu.sh

# 傳統 YOLOv7 方式（需要進入 yolov7 目錄）
cd yolov7
python train.py \
    --img 320 \
    --batch 128 \
    --epochs 300 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --data data/coco.yaml \
    --weights yolov7-tiny.pt \
    --device 0
```

## ⚙️ 參數調整

### 主要訓練參數

| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `--epochs` | 300 | 訓練 epoch 數 | 可調至 100-500 |
| `--batch-size` | 自動 | Batch size | 根據 VRAM 調整 |
| `--lr` | 自動 | 學習率 | 通常 0.01-0.1 |
| `--img` | 320 | 輸入圖片大小 | 固定 320 |
| `--device` | 0 | GPU 裝置 | 0,1,2,3... |
| `--workers` | 4 | DataLoader 工作數 | 根據 CPU 調整 |

### Batch Size 與學習率對照

根據不同 GPU 的建議配置：

| GPU | VRAM | Batch Size | Learning Rate | 指令 |
|-----|------|------------|---------------|------|
| RTX 4090 | 24GB | 128 | 0.01 | `--batch-size 128 --lr 0.01` |
| RTX 5090 | 32GB | 256 | 0.02 | `--batch-size 256 --lr 0.02` |
| H100 | 80GB | 640 | 0.05 | `--batch-size 640 --lr 0.05` |
| B200 | 192GB | 1536 | 0.12 | `--batch-size 1536 --lr 0.12` |

### 模型保存設定

YOLOv7 預設每 25 個 epoch 保存一次模型：

```yaml
# 在 yolov7/utils/general.py 中
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok | opt.exist_ok)  # 增加路徑
ckpt = {'epoch': epoch,
        'best_fitness': best_fitness,
        'training_results': results_file.read_text() if results_file.exists() else '',
        'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None if final_epoch else optimizer.state_dict(),
        'wandb_id': wandb_run.id if wandb else None}

# 保存檢查點
torch.save(ckpt, last)
if best_fitness == fi:
    torch.save(ckpt, best)
if (epoch > 0) & (epoch % 25 == 0):  # 每 25 epochs 保存
    torch.save(ckpt, w / f'epoch_{epoch}.pt')
```

修改保存頻率：

```bash
# 修改 yolov7/train.py 第 568 行附近
# 將 (epoch % 25 == 0) 改為你想要的間隔

# 每 10 epochs 保存：
if (epoch > 0) & (epoch % 10 == 0):
    torch.save(ckpt, w / f'epoch_{epoch}.pt')

# 每 50 epochs 保存：
if (epoch > 0) & (epoch % 50 == 0):
    torch.save(ckpt, w / f'epoch_{epoch}.pt')
```

### 超參數調整

主要超參數檔案：`yolov7/data/hyp.scratch.tiny.yaml`

```yaml
lr0: 0.01          # 初始學習率
lrf: 0.01          # 最終學習率 (lr0 * lrf)
momentum: 0.937    # SGD 動量
weight_decay: 0.0005  # 權重衰減
warmup_epochs: 3.0    # 預熱 epochs
warmup_momentum: 0.8  # 預熱動量
warmup_bias_lr: 0.1   # 預熱偏置學習率

# 資料增強
hsv_h: 0.015       # HSV-Hue 增強
hsv_s: 0.7         # HSV-Saturation 增強
hsv_v: 0.4         # HSV-Value 增強
degrees: 0.0       # 旋轉角度
translate: 0.1     # 平移比例
scale: 0.5         # 縮放比例
shear: 0.0         # 剪切角度
perspective: 0.0   # 透視變換
flipud: 0.0        # 垂直翻轉機率
fliplr: 0.5        # 水平翻轉機率
mosaic: 1.0        # Mosaic 增強機率
mixup: 0.05        # MixUp 增強機率
```

## 📊 訓練監控

### 背景驗證監控

```bash
# 啟動背景驗證腳本
bash tools/val_watcher.sh &

# 會每 60 秒檢查最新權重並執行驗證
# 結果存於 val_watcher.log
tail -f val_watcher.log
```

### 訓練日誌

```bash
# 查看訓練日誌
tail -f yolov7/runs/train/exp/results.txt

# 使用 TensorBoard
pip install tensorboard
tensorboard --logdir yolov7/runs/train/
```

## 🛠️ 工具集詳細說明

### 資料處理工具

#### 1. 生成資料清單 (`tools/gen_lists.py`)

```bash
# 基本用法（預設使用 data/coco）
python tools/gen_lists.py

# 完整參數
python tools/gen_lists.py \
    --coco-path ./data/coco \      # COCO 資料集路徑（預設：data/coco）
    --output-dir . \               # 輸出目錄（預設：專案根目錄）
    --skip-sha256 \                # 跳過 SHA256 生成
    --yolo-format                  # 生成 YOLOv7 格式清單

# 輸出檔案
# - train.txt (118,287 張訓練圖片路徑)
# - val.txt (5,000 張驗證圖片路徑)
# - calib.txt (500 張校正圖片路徑)
```

#### 2. 生成 SHA256 校驗 (`tools/gen_manifests.sh`)

```bash
# 為所有清單生成 SHA256 校驗檔
bash tools/gen_manifests.sh

# 輸出檔案
# - data/manifest_train.sha256
# - data/manifest_val.sha256
# - data/manifest_calib.sha256

# 驗證校驗和
cd data
sha256sum -c manifest_train.sha256
```

#### 3. COCO 資料集打包 (`tools/pack_coco.sh`)

```bash
# 互動式壓縮
bash tools/pack_coco.sh

# 選項：
# 1) tar.gz (Linux/Mac 推薦) - 約 18GB
# 2) zip (Windows 相容) - 約 19GB  
# 3) tar.xz (最高壓縮率) - 約 17GB

# 輸出：
# - data/coco.tar.gz (或 .zip/.tar.xz)
# - data/coco.tar.gz.md5 (校驗檔)
```

#### 4. COCO 資料集解包 (`tools/unpack_coco.sh`)

```bash
# 自動檢測並解壓
bash tools/unpack_coco.sh

# 指定檔案路徑
bash tools/unpack_coco.sh /path/to/coco.tar.gz

# 功能：
# - 自動檢測壓縮格式 (tar.gz/zip/tar.xz)
# - MD5 校驗（如果存在 .md5 檔）
# - 解壓進度顯示
# - 解壓後資料驗證
```

### 訓練相關工具

#### 5. 自適應訓練 (`tools/adaptive_train.py`)

```bash
# 自動模式（推薦）
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml

# 完整參數
python tools/adaptive_train.py \
    --img 320 \              # 輸入圖片大小
    --epochs 300 \           # 訓練輪數
    --device 0 \             # GPU 裝置
    --workers 4 \            # DataLoader 工作數
    --name adaptive \        # 實驗名稱
    --weights yolov7-tiny.pt # 預訓練權重

# 手動指定參數
python tools/adaptive_train.py \
    --batch-size 256 \       # 手動指定 batch size
    --lr 0.02 \              # 手動指定學習率
    --accumulate 2           # 梯度累積步數

# 功能：
# - 自動檢測 GPU VRAM 並選擇最佳 batch size
# - 根據 Linear Scaling Rule 計算學習率
# - 生成自定義超參數檔案
# - 支援手動覆蓋參數
```

#### 6. 背景驗證監控 (`tools/val_watcher.sh`)

```bash
# 啟動背景監控
bash tools/val_watcher.sh &

# 自定義參數
WATCH_DIR="yolov7/runs/train/exp" \
CHECK_INTERVAL=60 \
bash tools/val_watcher.sh &

# 功能：
# - 每 60 秒檢查新的權重檔案
# - 自動執行驗證 (test.py)
# - 記錄驗證結果到 val_watcher.log
# - 避免重複驗證同一檔案

# 查看監控日誌
tail -f val_watcher.log

# 停止監控
pkill -f val_watcher.sh
```

### 模型部署工具

#### 7. ONNX 模型評測 (`tools/eval_onnx.py`)

```bash
# 基本評測
python tools/eval_onnx.py \
    --model yolov7/model.onnx \
    --img 320 \
    --report eval_report.json

# 完整參數
python tools/eval_onnx.py \
    --model yolov7/model.onnx \     # ONNX 模型路徑
    --img 320 \                     # 輸入圖片大小
    --device cuda \                 # 推論裝置
    --batch 1 \                     # 批次大小
    --conf-thres 0.001 \            # 信心閾值
    --iou-thres 0.65 \              # NMS IoU 閾值
    --max-det 300 \                 # 最大檢測數
    --report eval_report.json \     # 評測報告路徑
    --append                        # 追加到現有報告

# 評測內容：
# - mAP50-95 (整體和各尺度)
# - 延遲測試 (100次推論的平均值)
# - 模型大小和參數量
# - 各類別的 AP 值
```

#### 8. INT8 PTQ 量化 (`tools/ort_ptq.py`)

```bash
# 基本量化
python tools/ort_ptq.py \
    --model yolov7/model.onnx \
    --calib data/calib.txt \
    --out yolov7/model-int8.onnx

# 完整參數
python tools/ort_ptq.py \
    --model yolov7/model.onnx \     # 輸入 ONNX 模型
    --calib data/calib.txt \        # 校正集清單
    --out yolov7/model-int8.onnx \  # 輸出量化模型
    --method percentile \           # 量化方法
    --percentile 99.99 \            # 百分位數閾值
    --batch-size 1 \                # 校正批次大小
    --img-size 320                  # 輸入圖片大小

# 量化方法選項：
# - percentile: 百分位數法 (推薦)
# - minmax: 最小最大值法
# - entropy: 熵校正法

# 量化配置：
# - Weights: INT8, symmetric, per-channel
# - Activations: INT8, asymmetric, per-tensor
# - 格式：QDQ (Quantize-Dequantize)
```

## 🔄 完整模型部署流程

### 步驟 1：訓練模型

```bash
# 使用自適應訓練（包含資料集檢查）
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml

# 或傳統方式
cd yolov7
python train.py --img 320 --batch 128 --epochs 300 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --data data/coco.yaml \
    --weights yolov7-tiny.pt
cd ..
```

### 步驟 2：導出 ONNX

```bash
cd yolov7

# 導出 ONNX
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --img 320 320 \
    --batch 1 \
    --include onnx \
    --opset 13

# 簡化模型
pip install onnxsim
python -m onnxsim model.onnx model-sim.onnx

cd ..
```

### 步驟 3：模型評測

```bash
# 評測原始 FP32 模型
python tools/eval_onnx.py \
    --model yolov7/model.onnx \
    --img 320 \
    --device cuda \
    --report eval_report.json

# 查看結果
cat eval_report.json | jq '.models[0]'
```

### 步驟 4：INT8 量化

```bash
# PTQ 量化
python tools/ort_ptq.py \
    --model yolov7/model.onnx \
    --calib data/calib.txt \
    --out yolov7/model-int8.onnx \
    --method percentile

# 生成 MD5 校驗
md5sum yolov7/model*.onnx > yolov7/onnx_md5.txt
```

### 步驟 5：量化模型評測

```bash
# 評測量化模型
python tools/eval_onnx.py \
    --model yolov7/model-int8.onnx \
    --img 320 \
    --device cuda \
    --report eval_report.json \
    --append

# 比較結果
cat eval_report.json | jq '.models[] | {name, mAP50_95, latency_ms}'
```

## ❓ 常見問題

### Q: 為什麼訓練時會報錯「找不到資料集」？

這是**正常的安全機制**。本專案禁用了 YOLOv7 的自動下載功能：

```bash
# 錯誤訊息範例：
# ❌ 錯誤：找不到以下資料集路徑：
#   - train: /path/to/data/coco/train2017
#   - val: /path/to/data/coco/val2017

# 解決方案：
# 1. 確認資料集在 data/coco/ 目錄
ls data/coco/
# 應該看到 train2017/, val2017/, annotations/

# 2. 生成資料清單
python tools/gen_lists.py

# 3. 重新開始訓練
python tools/adaptive_train.py --epochs 300 --data ../data/coco_local.yaml
```

### Q: 為什麼不能自動下載資料集？

理由：
1. **可控性**：避免意外的大量下載（COCO 約 20GB）
2. **安全性**：企業環境可能禁止外部下載
3. **可重現性**：確保使用相同版本的資料集
4. **效率**：避免重複下載已有的資料

### Q: 如何確認資料集路徑設定正確？

檢查 `data/coco_local.yaml` 的設定：

```yaml
# 正確設定（相對於 yolov7 工作目錄）
train: ../data/coco/train2017  # 專案根目錄/data/coco/train2017
val: ../data/coco/val2017      # 專案根目錄/data/coco/val2017

# 或使用清單檔案
train: ../train.txt   # 專案根目錄/train.txt
val: ../val.txt       # 專案根目錄/val.txt
```

### Q: 記憶體不足怎麼辦？

```bash
# 方法 1：降低 batch size
python tools/adaptive_train.py --batch-size 64

# 方法 2：使用梯度累積
python tools/adaptive_train.py --batch-size 32 --accumulate 4

# 方法 3：降低圖片解析度（不建議，影響精度）
python tools/adaptive_train.py --img 256
```

### Q: 訓練速度太慢？

```bash
# 方法 1：增加 workers
python tools/adaptive_train.py --workers 8

# 方法 2：使用多 GPU
./train_multi_gpu.sh

# 方法 3：使用 AMP（自動混合精度）
# 已預設啟用，無需額外設定
```

### Q: 如何恢復訓練？

```bash
# 從最後的檢查點恢復
cd yolov7
python train.py \
    --resume runs/train/exp/weights/last.pt
```

### Q: 如何修改類別數量？

```bash
# 1. 修改 yolov7/data/coco.yaml 中的 nc: 80
# 2. 修改 yolov7/cfg/training/yolov7-tiny.yaml 中的 nc: 80
# 3. 準備對應的資料集和標註
```

### Q: 訓練結果不佳？

檢查清單：
1. **資料集品質**：確認標註正確性
2. **學習率**：嘗試調整 `--lr` 參數
3. **資料增強**：檢查超參數設定
4. **預訓練權重**：確認使用正確的 `yolov7-tiny.pt`
5. **訓練時長**：增加 `--epochs` 數量

## 🎯 最佳實踐

1. **使用自動模式**：`adaptive_train.py` 會自動選擇最佳參數
2. **監控訓練**：使用 `val_watcher.sh` 監控驗證結果
3. **定期保存**：預設每 25 epochs 保存一次
4. **驗證設定**：訓練前確認資料集和環境
5. **資源監控**：監控 GPU 使用率和記憶體

## 🚀 進階功能與腳本

### 其他實用腳本

#### 簡化訓練腳本 (`train.sh`)

```bash
# 簡單的 YOLOv7 訓練包裝腳本
./train.sh

# 功能：
# - 自動切換到 yolov7 目錄
# - 使用預設參數啟動訓練
# - 處理路徑問題
```

#### 簡化測試腳本 (`test.sh`)

```bash
# 測試指定權重
./test.sh runs/train/exp/weights/best.pt

# 功能：
# - 自動切換到 yolov7 目錄
# - 使用標準 COCO 評測參數
# - 支援自定義權重路徑
```

#### 多 GPU 訓練腳本 (`train_multi_gpu.sh`)

```bash
# 自動檢測 GPU 數量並調整參數
./train_multi_gpu.sh

# 功能：
# - 自動檢測 GPU 數量
# - 根據 GPU 數量調整 batch size 和學習率
# - 使用 DDP (DistributedDataParallel)
# - 支援 1-8 GPU 配置
```

### 配置檔案

#### GPU 配置檔案 (`configs/gpu_profiles.yaml`)

```bash
# 查看 GPU 配置建議
cat configs/gpu_profiles.yaml

# 包含內容：
# - 各種 GPU 的最佳參數設定
# - 多 GPU 配置範例
# - 學習率調整策略
# - 記憶體優化建議
```

### 工具集總覽

| 工具 | 功能 | 主要用途 |
|------|------|----------|
| `gen_lists.py` | 生成資料清單 | 資料集準備 |
| `gen_manifests.sh` | 生成 SHA256 校驗 | 資料完整性驗證 |
| `pack_coco.sh` | 壓縮 COCO 資料集 | 資料集傳輸 |
| `unpack_coco.sh` | 解壓縮 COCO 資料集 | 資料集部署 |
| `adaptive_train.py` | 自適應參數訓練 | 智能訓練 |
| `val_watcher.sh` | 背景驗證監控 | 訓練監控 |
| `eval_onnx.py` | ONNX 模型評測 | 模型驗證 |
| `ort_ptq.py` | INT8 PTQ 量化 | 模型優化 |
| `train.sh` | 簡化訓練腳本 | 快速訓練 |
| `test.sh` | 簡化測試腳本 | 快速驗證 |
| `train_multi_gpu.sh` | 多 GPU 訓練 | 分散式訓練 |

## 📊 預期結果與基準

### 訓練預期結果

| 模型 | mAP50-95 | 訓練時間 | 模型大小 |
|------|----------|----------|----------|
| YOLOv7-tiny (FP16) | 33-35 | ~24小時 (RTX 4090) | 12MB |
| YOLOv7-tiny (INT8) | 31-34 | - | 6MB |

### 量化效果

| 量化方法 | 精度損失 | 推論加速 | 模型大小 |
|----------|----------|----------|----------|
| INT8 全量化 | -1.5~-3.0 mAP | 2-3× | 50% |
| INT8 混精度 | -0.5~-1.5 mAP | 1.5-2× | 60% |

### 不同平台推論速度 (320×320)

| GPU | FP16 (ms) | INT8 (ms) | 加速比 |
|-----|-----------|-----------|--------|
| RTX 4090 | 2.5 | 1.2 | 2.1× |
| RTX 5090 | 2.0 | 0.9 | 2.2× |
| H100 | 1.5 | 0.7 | 2.1× |
| B200 | 1.2 | 0.5 | 2.4× |

## 📝 輸出檔案結構

完整訓練後的檔案結構：

```
Yolov7Tiny320_Baseline/
├── data/
│   ├── coco/                    # COCO 資料集 (被 Git 忽略)
│   ├── train.txt               # 訓練清單 (118,287 張)
│   ├── val.txt                 # 驗證清單 (5,000 張)
│   ├── calib.txt              # 校正清單 (500 張)
│   ├── manifest_*.sha256       # SHA256 校驗檔
│   └── eval_report.json        # 評測報告
├── yolov7/
│   ├── runs/train/exp/
│   │   ├── weights/
│   │   │   ├── best.pt         # 最佳模型
│   │   │   ├── last.pt         # 最後模型
│   │   │   └── epoch_*.pt      # 定期保存 (每25輪)
│   │   ├── results.txt         # 訓練日誌
│   │   ├── hyp.yaml           # 超參數
│   │   └── opt.yaml           # 訓練選項
│   ├── model.onnx             # 導出的 ONNX 模型
│   ├── model-sim.onnx         # 簡化的 ONNX 模型
│   ├── model-int8.onnx        # INT8 量化模型
│   └── onnx_md5.txt          # ONNX 模型 MD5
├── tools/                      # 工具腳本集
├── configs/                    # 配置檔案
└── *.md                       # 說明文件
```

## 🔧 自定義與擴展

### 修改模型保存頻率

在 `yolov7/train.py` 中修改保存間隔：

```python
# 原始 (第 568 行附近)
if (epoch > 0) & (epoch % 25 == 0):  # 每 25 epochs

# 修改為每 10 epochs
if (epoch > 0) & (epoch % 10 == 0):

# 修改為每 50 epochs  
if (epoch > 0) & (epoch % 50 == 0):
```

### 自定義量化設定

修改 `tools/ort_ptq.py` 中的量化參數：

```python
# 量化配置
quant_format = QuantFormat.QDQ
quant_type_weights = QuantType.QInt8
quant_type_activations = QuantType.QUInt8

# 校正方法
calibration_method = CalibrationMethod.Percentile  # 或 MinMax, Entropy
```

### 添加新的評測指標

在 `tools/eval_onnx.py` 中可添加自定義評測指標。

## 📞 支援

如有問題，請：

1. 檢查本指南的[常見問題](#常見問題)部分
2. 查看專案的 GitHub Issues
3. 確認環境設定是否正確

---

**版本**: v1.0  
**更新日期**: 2025-08-11  
**相容性**: YOLOv7 官方版本