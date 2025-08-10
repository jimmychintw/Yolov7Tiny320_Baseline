# YOLOv7-tiny Baseline Makefile
# 根據 Baseline Spec v1.0 的自動化腳本

# 變數定義
PYTHON := python3
SHELL := /bin/bash
COCO_PATH := /data/coco
IMG_SIZE := 320
BATCH_SIZE := 128
EPOCHS := 300
DEVICE := 0
WORKERS := 4

# 路徑定義
YOLOV7_DIR := yolov7
TOOLS_DIR := tools
SCRIPTS_DIR := scripts
DATA_DIR := data
RUNS_DIR := runs

# 檔案路徑
TRAIN_LIST := train.txt
VAL_LIST := val.txt
CALIB_LIST := calib.txt
TRAIN2017_LIST := train2017.txt
VAL2017_LIST := val2017.txt

# 模型檔案
WEIGHTS := yolov7-tiny.pt
BEST_PT := $(YOLOV7_DIR)/runs/train/exp/weights/best.pt
MODEL_ONNX := model.onnx
MODEL_SIM_ONNX := model-sim.onnx
MODEL_INT8_ONNX := model-int8.onnx

# 顏色定義
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: all help setup data train export_onnx ptq eval clean baseline

# 預設目標
all: baseline

# 說明
help:
	@echo "$(GREEN)YOLOv7-tiny Baseline Makefile$(NC)"
	@echo ""
	@echo "可用目標:"
	@echo "  $(YELLOW)baseline$(NC)      - 執行完整 baseline 流程"
	@echo "  $(YELLOW)setup$(NC)         - 設定環境和下載權重"
	@echo "  $(YELLOW)data$(NC)          - 生成資料清單和校驗檔"
	@echo "  $(YELLOW)train$(NC)         - 訓練模型 (300 epochs)"
	@echo "  $(YELLOW)export_onnx$(NC)   - 導出 ONNX 模型"
	@echo "  $(YELLOW)ptq$(NC)           - 執行 PTQ 量化"
	@echo "  $(YELLOW)eval$(NC)          - 評測模型"
	@echo "  $(YELLOW)clean$(NC)         - 清理生成檔案"
	@echo ""
	@echo "參數:"
	@echo "  COCO_PATH=$(COCO_PATH)"
	@echo "  IMG_SIZE=$(IMG_SIZE)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  DEVICE=$(DEVICE)"

# 完整 baseline 流程
baseline: setup data train export_onnx ptq eval report
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Baseline 流程完成！$(NC)"
	@echo "$(GREEN)========================================$(NC)"

# 環境設定
setup:
	@echo "$(GREEN)=== 環境設定 ===$(NC)"
	# 檢查 YOLOv7 目錄
	@if [ ! -d "$(YOLOV7_DIR)" ]; then \
		echo "$(RED)錯誤: 請先執行 git clone https://github.com/WongKinYiu/yolov7$(NC)"; \
		exit 1; \
	fi
	# 下載預訓練權重
	@if [ ! -f "$(YOLOV7_DIR)/$(WEIGHTS)" ]; then \
		echo "下載 YOLOv7-tiny 權重..."; \
		cd $(YOLOV7_DIR) && \
		wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt; \
	else \
		echo "權重已存在: $(YOLOV7_DIR)/$(WEIGHTS)"; \
	fi
	# 安裝相依套件
	@echo "安裝 Python 套件..."
	@pip install -r requirements.txt 2>/dev/null || true
	@echo "$(GREEN)✓ 環境設定完成$(NC)"

# 生成資料清單
data:
	@echo "$(GREEN)=== 生成資料清單 ===$(NC)"
	@$(PYTHON) $(TOOLS_DIR)/gen_lists.py \
		--coco-path $(COCO_PATH) \
		--output-dir . \
		--yolo-format
	@echo "$(GREEN)✓ 資料清單生成完成$(NC)"

# 訓練模型
train:
	@echo "$(GREEN)=== 開始訓練 ===$(NC)"
	@echo "參數:"
	@echo "  影像尺寸: $(IMG_SIZE)"
	@echo "  批次大小: $(BATCH_SIZE)"
	@echo "  Epochs: $(EPOCHS)"
	@echo "  裝置: cuda:$(DEVICE)"
	# 使用訓練包裝腳本
	@$(PYTHON) $(SCRIPTS_DIR)/train_baseline.py \
		--img-size $(IMG_SIZE) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--data-path . \
		--save-period 25
	@echo "$(GREEN)✓ 訓練完成$(NC)"

# 直接使用 YOLOv7 原生訓練（備選）
train-native:
	@echo "$(GREEN)=== YOLOv7 原生訓練 ===$(NC)"
	cd $(YOLOV7_DIR) && \
	$(PYTHON) train.py \
		--img $(IMG_SIZE) \
		--batch $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--cfg cfg/training/yolov7-tiny.yaml \
		--data ../$(DATA_DIR)/coco_baseline.yaml \
		--weights $(WEIGHTS) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--hyp ../$(DATA_DIR)/hyp_baseline.yaml \
		--name baseline \
		--project runs/train

# 導出 ONNX
export_onnx:
	@echo "$(GREEN)=== 導出 ONNX 模型 ===$(NC)"
	@if [ ! -f "$(BEST_PT)" ]; then \
		echo "$(RED)錯誤: 找不到 best.pt，請先訓練模型$(NC)"; \
		exit 1; \
	fi
	cd $(YOLOV7_DIR) && \
	$(PYTHON) export.py \
		--weights $(BEST_PT) \
		--img $(IMG_SIZE) $(IMG_SIZE) \
		--batch 1 \
		--grid \
		--simplify \
		--include onnx
	# 移動並重命名 ONNX 檔案
	@mv $(YOLOV7_DIR)/runs/train/exp/weights/best.onnx $(MODEL_ONNX) 2>/dev/null || true
	# 簡化 ONNX
	@$(PYTHON) -m onnxsim $(MODEL_ONNX) $(MODEL_SIM_ONNX)
	# 生成 MD5
	@md5sum $(MODEL_ONNX) $(MODEL_SIM_ONNX) > onnx_md5.txt
	@echo "$(GREEN)✓ ONNX 導出完成$(NC)"

# PTQ 量化
ptq:
	@echo "$(GREEN)=== PTQ 量化 ===$(NC)"
	@if [ ! -f "$(MODEL_ONNX)" ]; then \
		echo "$(RED)錯誤: 找不到 model.onnx，請先導出 ONNX$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(CALIB_LIST)" ]; then \
		echo "$(RED)錯誤: 找不到 calib.txt，請先生成資料清單$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON) $(TOOLS_DIR)/ort_ptq.py \
		--model $(MODEL_ONNX) \
		--calib $(CALIB_LIST) \
		--out $(MODEL_INT8_ONNX) \
		--method percentile \
		--percentile 99.99 \
		--img-size $(IMG_SIZE)
	@md5sum $(MODEL_INT8_ONNX) > onnx_int8_md5.txt
	@echo "$(GREEN)✓ PTQ 量化完成$(NC)"

# 評測模型
eval:
	@echo "$(GREEN)=== 模型評測 ===$(NC)"
	# 評測 FP32 模型
	@if [ -f "$(MODEL_ONNX)" ]; then \
		echo "評測 FP32 模型..."; \
		$(PYTHON) $(TOOLS_DIR)/eval_onnx.py \
			--model $(MODEL_ONNX) \
			--val-list $(VAL_LIST) \
			--img $(IMG_SIZE) \
			--report eval_report.json; \
	fi
	# 評測 INT8 模型
	@if [ -f "$(MODEL_INT8_ONNX)" ]; then \
		echo "評測 INT8 模型..."; \
		$(PYTHON) $(TOOLS_DIR)/eval_onnx.py \
			--model $(MODEL_INT8_ONNX) \
			--val-list $(VAL_LIST) \
			--img $(IMG_SIZE) \
			--report eval_report.json \
			--append; \
	fi
	@echo "$(GREEN)✓ 評測完成$(NC)"

# 生成報告
report:
	@echo "$(GREEN)=== 生成報告 ===$(NC)"
	@echo "交付檔案清單:"
	@echo "  ✓ best.pt (訓練權重)"
	@echo "  ✓ model.onnx (FP32 ONNX)"
	@echo "  ✓ model-sim.onnx (簡化 ONNX)"
	@echo "  ✓ model-int8.onnx (INT8 ONNX)"
	@echo "  ✓ onnx_md5.txt (MD5 校驗)"
	@echo "  ✓ onnx_int8_md5.txt (INT8 MD5)"
	@echo "  ✓ train.txt, val.txt, calib.txt (資料清單)"
	@echo "  ✓ manifest_*.sha256 (SHA256 校驗)"
	@echo "  ✓ eval_report.json (評測報告)"
	@if [ -f "eval_report.json" ]; then \
		echo ""; \
		echo "評測結果摘要:"; \
		$(PYTHON) -c "import json; \
			r = json.load(open('eval_report.json')); \
			for model, data in r.items(): \
				m = data.get('metrics', {}); \
				l = data.get('latency', {}); \
				print(f'  {model}:'); \
				print(f'    mAP50-95: {m.get(\"mAP50-95\", 0):.3f}'); \
				print(f'    mAP50: {m.get(\"mAP50\", 0):.3f}'); \
				print(f'    延遲 (median): {l.get(\"median\", 0):.2f} ms'); \
				print(f'    延遲 (P95): {l.get(\"p95\", 0):.2f} ms')"; \
	fi
	@echo "$(GREEN)✓ 報告生成完成$(NC)"

# 生成環境記錄
env:
	@echo "$(GREEN)=== 生成環境記錄 ===$(NC)"
	@echo "# 環境版本記錄" > ENV.md
	@echo "" >> ENV.md
	@echo "生成時間: $$(date '+%Y-%m-%d %H:%M:%S')" >> ENV.md
	@echo "" >> ENV.md
	@echo "## Python 套件版本" >> ENV.md
	@echo "\`\`\`" >> ENV.md
	@echo "Python: $$(python --version)" >> ENV.md
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)')" >> ENV.md
	@echo "torchvision: $$(python -c 'import torchvision; print(torchvision.__version__)')" >> ENV.md
	@echo "ONNX: $$(python -c 'import onnx; print(onnx.__version__)')" >> ENV.md
	@echo "ONNXRuntime: $$(python -c 'import onnxruntime; print(onnxruntime.__version__)')" >> ENV.md
	@echo "NumPy: $$(python -c 'import numpy; print(numpy.__version__)')" >> ENV.md
	@echo "OpenCV: $$(python -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not installed')" >> ENV.md
	@echo "\`\`\`" >> ENV.md
	@echo "" >> ENV.md
	@echo "## CUDA 版本" >> ENV.md
	@echo "\`\`\`" >> ENV.md
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> ENV.md 2>/dev/null || echo "No CUDA device" >> ENV.md
	@echo "\`\`\`" >> ENV.md
	@echo "" >> ENV.md
	@echo "## YOLOv7 Commit" >> ENV.md
	@echo "\`\`\`" >> ENV.md
	@cd $(YOLOV7_DIR) && git rev-parse HEAD >> ../ENV.md 2>/dev/null || echo "Not a git repo" >> ../ENV.md
	@echo "\`\`\`" >> ENV.md
	@echo "$(GREEN)✓ ENV.md 已生成$(NC)"

# 清理
clean:
	@echo "$(YELLOW)清理生成檔案...$(NC)"
	@rm -f $(TRAIN_LIST) $(VAL_LIST) $(CALIB_LIST)
	@rm -f $(TRAIN2017_LIST) $(VAL2017_LIST)
	@rm -f manifest_*.sha256
	@rm -f *.onnx *.txt
	@rm -f eval_report.json
	@rm -rf $(RUNS_DIR)
	@rm -rf val_results
	@rm -f .val_watcher_processed
	@rm -f val_watcher.log
	@echo "$(GREEN)✓ 清理完成$(NC)"

# 深度清理（包括訓練結果）
clean-all: clean
	@echo "$(YELLOW)深度清理...$(NC)"
	@rm -rf $(YOLOV7_DIR)/runs
	@rm -f $(YOLOV7_DIR)/*.pt
	@rm -f ENV.md
	@echo "$(GREEN)✓ 深度清理完成$(NC)"

# 檢查狀態
status:
	@echo "$(GREEN)=== 專案狀態 ===$(NC)"
	@echo "YOLOv7 目錄: $$([ -d $(YOLOV7_DIR) ] && echo '✓' || echo '✗')"
	@echo "預訓練權重: $$([ -f $(YOLOV7_DIR)/$(WEIGHTS) ] && echo '✓' || echo '✗')"
	@echo "訓練清單: $$([ -f $(TRAIN_LIST) ] && echo '✓' || echo '✗')"
	@echo "驗證清單: $$([ -f $(VAL_LIST) ] && echo '✓' || echo '✗')"
	@echo "校正清單: $$([ -f $(CALIB_LIST) ] && echo '✓' || echo '✗')"
	@echo "最佳權重: $$([ -f $(BEST_PT) ] && echo '✓' || echo '✗')"
	@echo "FP32 ONNX: $$([ -f $(MODEL_ONNX) ] && echo '✓' || echo '✗')"
	@echo "INT8 ONNX: $$([ -f $(MODEL_INT8_ONNX) ] && echo '✓' || echo '✗')"
	@echo "評測報告: $$([ -f eval_report.json ] && echo '✓' || echo '✗')"

# 測試單一步驟
test-data:
	@$(MAKE) data

test-ptq:
	@$(MAKE) ptq

test-eval:
	@$(MAKE) eval

.DEFAULT_GOAL := help