# 資料準備指南

## 資料目錄結構

```
data/
├── train.txt           # 訓練集圖片路徑清單 (118,287 張)
├── val.txt            # 驗證集圖片路徑清單 (5,000 張)  
├── calib.txt          # 量化校正集清單 (500 張，從 val 抽樣)
├── manifest_train.sha256  # 訓練集 SHA256 校驗
├── manifest_val.sha256    # 驗證集 SHA256 校驗
└── manifest_calib.sha256  # 校正集 SHA256 校驗
```

## 生成步驟

### 1. 準備 COCO 資料集
確保 COCO 2017 資料集已下載到 `/data/coco/`：
```bash
/data/coco/
├── train2017/     # 訓練圖片
├── val2017/       # 驗證圖片
└── annotations/   # 標註檔案
```

### 2. 生成資料清單
```bash
# 生成 train.txt, val.txt, calib.txt
python tools/gen_lists.py --coco-path /data/coco --output-dir ./data

# 如果需要 YOLOv7 格式清單
python tools/gen_lists.py --coco-path /data/coco --output-dir ./data --yolo-format
```

### 3. 生成 SHA256 校驗檔
```bash
# 方法一：使用 shell 腳本
bash tools/gen_manifests.sh

# 方法二：使用 Python (包含在 gen_lists.py 中)
python tools/gen_lists.py --coco-path /data/coco --output-dir ./data
```

### 4. 驗證檔案
```bash
# 檢查生成的檔案
ls -lh data/

# 驗證檔案數量
wc -l data/*.txt

# 預期結果：
# train.txt: 118,287 行
# val.txt: 5,000 行  
# calib.txt: 500 行
```

## 檔案說明

### train.txt / val.txt
- 每行一個圖片絕對路徑
- 格式：`/data/coco/train2017/000000xxxxx.jpg`
- 按檔名排序確保可重現性

### calib.txt
- 從驗證集穩定抽樣（每 10 張取 1 張）
- 最多 512 張圖片
- 用於 INT8 量化校正

### manifest_*.sha256
- SHA256 校驗和檔案
- 格式：`<sha256sum> <file_path>`
- 用於驗證資料完整性

## 注意事項

1. **路徑必須是絕對路徑**：所有清單檔案中的路徑都使用絕對路徑
2. **排序保證可重現**：使用 `sorted()` 確保檔案順序一致
3. **校正集抽樣固定**：使用固定間隔抽樣（每 10 張取 1 張）而非隨機
4. **SHA256 用於驗證**：可用於檢查資料集是否被修改過