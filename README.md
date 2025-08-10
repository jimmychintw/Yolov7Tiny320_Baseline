# YOLOv7-tiny Baseline (320×320)

![YOLOv7](https://img.shields.io/badge/YOLOv7-tiny-blue)
![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![ONNX](https://img.shields.io/badge/ONNX-1.15.0-red)

基於 **Baseline Spec v1.0** 的 YOLOv7-tiny 標準化實作，專注於在 COCO2017 資料集上進行 320×320 輸入尺寸的訓練與量化。

## 🎯 專案目標

建立一個可重現、標準化的 YOLOv7-tiny baseline，用於：
- 模型架構改進的公平對照
- 跨平台部署的一致性驗證  
- 量化效果的準確評估

## 📋 技術規格

- **模型**: YOLOv7-tiny
- **資料集**: COCO 2017 (官方 split)
- **輸入尺寸**: 320×320 (letterbox)
- **訓練**: AMP (FP16 混合精度)，300 epochs
- **量化**: PTQ (ONNX Runtime 靜態量化，QDQ 格式)
- **評測標準**: COCO mAP50-95 (含 S/M/L)

## 🚀 快速開始

### 1. 環境設定
```bash
# 克隆倉庫（包含子模組）
git clone --recursive https://github.com/jimmychintw/Yolov7Tiny320_Baseline.git
cd Yolov7Tiny320_Baseline

# 如果忘記使用 --recursive，可執行：
# git submodule update --init --recursive

# 一鍵設定環境
./scripts/setup_env.sh

# 啟用環境
source activate_baseline.sh
```

### 2. 完整 Baseline 流程
```bash
# 執行完整 baseline 流程
make baseline
```

### 3. 分步執行
```bash
make help          # 查看所有可用命令
make setup          # 環境設定和權重下載
make data           # 生成資料清單和校驗檔
make train          # 訓練模型 (300 epochs)
make export_onnx    # 導出 ONNX 模型
make ptq            # PTQ 量化
make eval           # 評測模型
make status         # 檢查專案狀態
```

## 🏗️ 專案架構

```
.
├── CLAUDE.md                    # Claude Code 指引文檔
├── Makefile                     # 自動化流程
├── requirements.txt             # Python 套件需求
├── scripts/
│   ├── setup_env.sh            # 環境設定腳本
│   └── train_baseline.py       # 訓練包裝腳本
├── tools/
│   ├── gen_lists.py            # 資料清單生成
│   ├── ort_ptq.py              # ONNX Runtime PTQ 量化
│   ├── eval_onnx.py            # ONNX 模型評測
│   └── val_watcher.sh          # 背景驗證監控
├── yolov7/                     # YOLOv7 官方程式碼
└── data/                       # 資料配置檔案
```

## 🔧 主要功能

### 決定性訓練
- 固定隨機種子 (42)
- CUDA 決定性模式
- 環境變數自動設定

### 自動化監控
- 背景驗證每 60 秒檢查新 checkpoint
- mAP 歷史記錄追蹤
- 自動生成詳細報告

### 量化支援
- ONNX Runtime PTQ
- QDQ 格式輸出
- Percentile 99.99% 校正
- INT8 對 FP16 一致性驗證

### Baseline 規範遵循
- **Epoch 0-260**: Mosaic/MixUp 資料增強開啟
- **Epoch 261-300**: 關閉 Mosaic/MixUp 進行收尾
- 嚴格的前處理參數 (letterbox, normalize)
- 固定的 NMS 參數 (conf=0.001, iou=0.65, max_det=300)

## 📊 預期結果

| 模型 | mAP50-95 | 延遲 (INT8) | 檔案大小 |
|------|----------|-------------|----------|
| FP16 | 33-35 | - | ~12MB |
| INT8 全量化 | 30-33 | 0.5-0.7× | ~6MB |
| INT8 混精度 | 32-35 | 0.6-0.8× | ~8MB |

## 📦 交付物清單

完成 baseline 後會產生以下檔案：
- ✅ `best.pt` - 訓練最佳權重 (AMP)
- ✅ `model.onnx` / `model-sim.onnx` - 原始與簡化 ONNX
- ✅ `model-int8.onnx` - INT8 量化模型
- ✅ `onnx_md5.txt` / `onnx_int8_md5.txt` - MD5 校驗檔
- ✅ `train.txt` / `val.txt` / `calib.txt` - 資料清單
- ✅ `manifest_*.sha256` - SHA256 校驗檔
- ✅ `eval_report.json` - 評測報告
- ✅ `ENV.md` - 環境版本記錄

## ⚙️ 環境需求

### 硬體需求
- **GPU**: NVIDIA GPU with CUDA 12.1 (建議)
- **Memory**: 至少 8GB GPU 記憶體
- **Storage**: 至少 50GB 可用空間

### 軟體需求
- **Python**: 3.8+
- **PyTorch**: 2.2.2
- **CUDA**: 12.1 (建議，可使用其他版本但可能影響重現性)

### 主要相依套件
```
torch==2.2.2
torchvision==0.17.2
onnx==1.15.0
onnxruntime==1.17.1
onnxsim==0.4.36
pycocotools>=2.0.6
```

## 🔄 工作流程

1. **資料準備**: 生成 COCO2017 訓練/驗證/校正集清單
2. **模型訓練**: AMP 混合精度訓練 300 epochs
3. **ONNX 導出**: 導出並簡化 ONNX 模型
4. **PTQ 量化**: 使用 512 張校正影像進行量化
5. **模型評測**: mAP 評估和延遲測試
6. **報告生成**: 產生完整的評測報告和環境記錄

## 📝 使用範例

### 訓練自定義配置
```bash
# 使用自定義批次大小和 epochs
python scripts/train_baseline.py \
    --batch-size 64 \
    --epochs 200 \
    --img-size 416 \
    --device 0
```

### 量化特定模型
```bash
python tools/ort_ptq.py \
    --model model.onnx \
    --calib calib.txt \
    --out custom-int8.onnx \
    --method percentile \
    --percentile 99.9
```

### 評測模型效能
```bash
python tools/eval_onnx.py \
    --model model-int8.onnx \
    --val-list val.txt \
    --img 320 \
    --report results.json
```

## 🤝 貢獻指南

1. Fork 此倉庫
2. 建立功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 開啟 Pull Request

## 📄 授權

此專案基於 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) - YOLOv7 官方實作
- [COCO Dataset](https://cocodataset.org/) - 評測資料集
- [ONNX Runtime](https://onnxruntime.ai/) - 模型量化和推論

## 📧 聯絡方式

如有問題或建議，請開啟 [Issue](https://github.com/your-username/Yolov7Tiny320_Baseline/issues)

---

## 🔗 相關連結

- [YOLOv7 論文](https://arxiv.org/abs/2207.02696)
- [ONNX Runtime 文檔](https://onnxruntime.ai/docs/)
- [PyTorch 官方網站](https://pytorch.org/)
- [COCO 評測指標](https://cocodataset.org/#detection-eval)