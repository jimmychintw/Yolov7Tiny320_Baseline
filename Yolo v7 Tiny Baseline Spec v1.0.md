下面是一份可直接採用的**Baseline 定義文檔（v1.0）**。之後你任何模型改動，都以此為對照；資料、流程與評測全鎖定，確保公平可重現。





# **Baseline Spec — YOLOv7‑tiny (AMP) on COCO2017 320×320 + PTQ**







## **0. 目標與範圍**





- **Model**: YOLOv7‑tiny
- **Dataset**: COCO 2017（官方 split），輸入統一 **320×320**（letterbox）
- **Training**: **AMP（FP16 混合精度）**，**300 epochs**，不中斷（可自動每 25 epoch 抽測）
- **Quantization**: **PTQ（ONNX Runtime 靜態量化，QDQ）**；各部署平台**一律**由該 INT8 ONNX 再轉
- **Evaluation**: COCO mAP50‑95（含 S/M/L），固定前處理與 NMS
- **Cross‑platform 容忍度**：各平台對「黃金引擎（ORT‑CPU）」的 mAP50‑95 差 ≤ **0.5**





------





## **1. 環境與版本（鎖定）**





- **CUDA**: 12.1、**PyTorch**: 2.2.2、**torchvision**: 0.17.2
- **onnx**: 1.15.0、**onnxruntime**: 1.17.1、**onnxruntime‑tools**: 1.7.0、**onnxsim**: 0.4.36
- **YOLOv7 repo**: 固定到指定 commit（記錄於 ENV.md）
- **決定性設定（盡可能）**：



```
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=0
python - <<'PY'
import torch, random, numpy as np
random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark=False
PY
```



- > AMP 仍可能有 ±0.1 mAP 內微小抖動，可接受做為 baseline。





------





## **2. 數據與切分（鎖定）**





- **路徑**：/data/coco/{train2017,val2017,annotations}
- **影像清單（固定順序）**：



```
find /data/coco/train2017 -type f -name "*.jpg" | sort > train.txt
find /data/coco/val2017   -type f -name "*.jpg" | sort > val.txt
# 校正集：val 中穩定抽樣 512 張
awk 'NR%10==1' val.txt | head -n 512 > calib.txt
# 產生 SHA256 清單
xargs -a train.txt -I{} sha256sum "{}" > manifest_train.sha256
xargs -a val.txt   -I{} sha256sum "{}" > manifest_val.sha256
xargs -a calib.txt -I{} sha256sum "{}" > manifest_calib.sha256
```



- 
- **前處理（訓練/驗證一致）**：letterbox auto=False, scaleFill=False, scaleup=True, color=(114,114,114), stride=32，normalize(1/255)





------





## **3. 訓練規格（AMP 300E）**





- **img**: 320；**epochs**: 300（基準不早停）

- **batch**: 128（顯存不足 → batch=64 + grad accumulation=2，保持等效 128）

- **optimizer**: SGD（等效 batch 64 時 lr0=0.01；用線性縮放），lrf=0.01（cosine decay）、momentum=0.937、weight_decay=5e-4

- **AMP**: 開（autocast+GradScaler）；**EMA**: 開

- **資料增強**：Epoch 0–260：開 **Mosaic/MixUp/HSV/affine**；Epoch 261–300：**關閉 Mosaic/MixUp** 收尾

- **Label smoothing**: 0.05；**workers**: 4（固定）

- **模型保存/抽測**：每 **25 epoch** 保存 last.pt，並啟動**背景驗證**（不中斷訓練）

  

  - 訓練啟動（示例）：

  



```
python train.py --img 320 --batch 128 --epochs 300 \
  --data data/coco.yaml --weights yolov7-tiny.pt \
  --device 0 --workers 4 --amp --save-period 25
```



- 

  - 
  - 後台抽測（watcher，示例）：

  



```
# 每 60 秒檢查一次是否有新的 last.pt，找到就跑 val（單獨進程）
while true; do
  L=./runs/train/exp/weights/last.pt
  [ -f "$L" ] && python val.py --weights $L --data data/coco.yaml --img 320 --conf-thres 0.001 --iou-thres 0.65 --max-det 300
  sleep 60
done
```



- 

  > 若原版 --save-period 不支援，可在訓練程式中以 if (epoch+1)%25==0: save_ckpt() 實作；抽測一律獨立進程。





------





## **4. 導出（ONNX）**





- **opset**: 13（主力），另存 17 備選
- **shape**: 靜態 1×3×320×320；**simplify**：原始與簡化版都保存 md5



```
python export.py --weights runs/train/exp/weights/best.pt \
  --img 320 320 --batch 1 --include onnx --opset 13
python -m onnxsim model.onnx model-sim.onnx
md5sum model.onnx model-sim.onnx > onnx_md5.txt
```





------





## **5. PTQ（黃金 INT8 模型，ONNX Runtime）**





- **Quant 格式**：QDQ
- **Weights**：INT8，**symmetric, per‑channel**
- **Activations**：INT8，**asymmetric, per‑tensor**
- **Calib**：**Percentile 99.99%**（或 MinMax；擇一固定）
- **校正集**：calib.txt（512 張；無增強；與前處理一致）
- **輸出**：model-int8.onnx + md5





> 後續 **TensorRT/OpenVINO/NPU** 等平台一律以 model-int8.onnx 為**唯一來源**轉換，**不得**各自重跑校正。



------





## **6. 評測規格（統一）**





- **前處理**：同訓練的 letterbox，normalize(1/255)
- **NMS/閾值**：conf_thres=0.001, iou_thres=0.65, max_det=300, class_agnostic=False
- **指標**：COCO mAP50‑95（含 S/M/L 分項）；pycocotools 版本鎖定
- **延遲**：單 batch=1；**預熱 50 次**，連跑 **200 次**取**中位數**與 **P95**；同機同 power profile
- **黃金引擎**：ONNX Runtime **CPU**（model-int8.onnx）的 mAP 作為數值基準





------





## **7. 驗收門檻與預期（Sanity 範圍）**





- **mAP50‑95（AMP/FP16）**：~ **33–35**（COCO val2017, 320×320, YOLOv7‑tiny）
- **INT8（全量化）對 FP16 的掉點**：**‑1.5 ~ ‑3.0 mAP**
- **INT8（層級混精：首層 + Detect head 回 FP16）**：通常可收斂到 **‑0 ~ ‑1.5 mAP**
- **延遲**：INT8 約 FP16 的 **0.5–0.7×**（依硬體而異）
- **跨平台一致性**：各引擎對 ORT‑CPU 的差 ≤ **0.5 mAP**





------





## **8. 交付物（每次實驗均需）**





- best.pt（AMP）
- model.onnx、model-sim.onnx、model-int8.onnx、onnx_md5.txt、onnx_int8_md5.txt
- train.txt、val.txt、calib.txt、manifest_*.sha256
- eval_report.json（mAP50‑95/50/S/M/L、延遲中位數/P95、平台對照）
- ENV.md（CUDA/PyTorch/ONNX/ORT/pycocotools 版本與 YOLOv7 commit）
- （若改動訓練程式）patch.diff 或 fork commit id





------





## **9. 變更管理與公平對照**





- **嚴禁改動**：資料與 split、前處理、訓練 epoch、AMP 開關、PTQ 食譜、校正集、NMS 參數。
- **允許改動**：模型架構、loss、增強策略（但**收尾關閉 Mosaic/MixUp 的節點不變**）。
- **報表**：一律呈現 ΔmAP50‑95、ΔS‑mAP、延遲中位數與 P95 的變化；INT8 以相同 model-int8.onnx 流程產生。





------





## **10. 附：一鍵自動化（建議 Make 目標）**



```
baseline:
\tpython tools/gen_lists.py  # 生成 train/val/calib + manifests
\tpython train.py --img 320 --batch 128 --epochs 300 --amp --save-period 25 ...
\tnice -n 10 bash tools/val_watcher.sh &  # 背景抽測

export_onnx:
\tpython export.py --weights runs/train/exp/weights/best.pt --img 320 320 --batch 1 --include onnx --opset 13
\tpython -m onnxsim model.onnx model-sim.onnx
\tmd5sum model.onnx model-sim.onnx > onnx_md5.txt

ptq:
\tpython tools/ort_ptq.py --model model.onnx --calib calib.txt --out model-int8.onnx --method percentile
\tmd5sum model-int8.onnx > onnx_int8_md5.txt

eval:
\tpython tools/eval_onnx.py --model model.onnx --img 320 --report eval_report.json
\tpython tools/eval_onnx.py --model model-int8.onnx --img 320 --report eval_report.json --append
```



