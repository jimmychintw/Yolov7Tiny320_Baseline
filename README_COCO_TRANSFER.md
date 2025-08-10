# COCO 資料集傳輸指南

## 快速開始

### 在源機器上（有 COCO 資料集）

```bash
# 壓縮 COCO 資料集
bash tools/pack_coco.sh

# 選擇壓縮格式：
# 1) tar.gz - Linux/Mac 推薦（約 18GB）
# 2) zip - Windows 相容（約 19GB）  
# 3) tar.xz - 最高壓縮率（約 17GB，但較慢）

# 產生檔案：
# - data/coco.tar.gz (或 .zip/.tar.xz)
# - data/coco.tar.gz.md5 (校驗檔)
```

### 傳輸到目標機器

```bash
# 方法 1：SCP（簡單）
scp data/coco.tar.gz user@remote:/path/to/project/data/

# 方法 2：rsync（支援續傳，推薦大檔案）
rsync -avP data/coco.tar.gz user@remote:/path/to/project/data/

# 方法 3：分割傳輸（網路不穩定時）
split -b 1G data/coco.tar.gz coco.tar.gz.part.
scp coco.tar.gz.part.* user@remote:/path/to/project/data/
# 在遠端合併：cat coco.tar.gz.part.* > coco.tar.gz

# 方法 4：使用雲端儲存
# 上傳到 Google Drive/Dropbox/S3 等，再下載
```

### 在目標機器上（git clone 後）

```bash
# 1. Clone 專案
git clone https://github.com/your-repo/Yolov7Tiny320_Baseline.git
cd Yolov7Tiny320_Baseline

# 2. 將 coco.tar.gz 放入 data/ 目錄
mv ~/Downloads/coco.tar.gz data/

# 3. 解壓縮
bash tools/unpack_coco.sh

# 4. 生成資料清單
python tools/gen_lists.py --coco-path ./data/coco --output-dir ./data

# 5. 開始訓練
python tools/adaptive_train.py --epochs 300
```

## 詳細說明

### 壓縮檔大小預估

| 格式 | 壓縮率 | 檔案大小 | 壓縮時間 | 解壓時間 |
|------|--------|----------|----------|----------|
| tar.gz | 中等 | ~18GB | 10-15分 | 5-10分 |
| zip | 較低 | ~19GB | 8-12分 | 5-8分 |
| tar.xz | 最高 | ~17GB | 20-30分 | 10-15分 |

原始 COCO 資料集：約 20GB（118,287 + 5,000 張圖片）

### 完整工作流程

```bash
# === 機器 A（源）===
cd Yolov7Tiny320_Baseline

# 檢查資料集
ls -la data/coco/
# train2017/ val2017/ annotations/

# 壓縮
bash tools/pack_coco.sh
# 選擇 1 (tar.gz)

# 檢查輸出
ls -lh data/coco.tar.gz*
# coco.tar.gz (18GB)
# coco.tar.gz.md5

# === 傳輸 ===
scp data/coco.tar.gz user@machine-b:/tmp/

# === 機器 B（目標）===
git clone https://github.com/xxx/Yolov7Tiny320_Baseline.git
cd Yolov7Tiny320_Baseline

# 移動壓縮檔
mv /tmp/coco.tar.gz data/

# 解壓縮
bash tools/unpack_coco.sh

# 驗證
ls data/coco/
# train2017/ val2017/ annotations/

# 生成清單
python tools/gen_lists.py --coco-path ./data/coco --output-dir ./data --skip-sha256

# 確認
wc -l data/*.txt
# 118287 data/train.txt
# 5000 data/val.txt
# 500 data/calib.txt

# 準備完成！
```

### 故障排除

#### 問題 1：壓縮檔損壞
```bash
# 驗證 MD5
md5sum -c data/coco.tar.gz.md5

# 如果失敗，重新傳輸
```

#### 問題 2：空間不足
```bash
# 檢查空間（需要約 40GB：20GB 資料 + 20GB 壓縮檔）
df -h .

# 解壓後刪除壓縮檔
bash tools/unpack_coco.sh
# 選擇 "y" 刪除壓縮檔
```

#### 問題 3：解壓縮失敗
```bash
# 手動解壓
cd data
tar -xzf coco.tar.gz

# 或使用 pv 顯示進度
pv coco.tar.gz | tar -xz
```

### 自動化腳本

建立 `setup_coco.sh`：

```bash
#!/bin/bash
# 一鍵設定 COCO 資料集

# 下載（如果有雲端連結）
# wget https://your-storage/coco.tar.gz -P data/

# 解壓縮
bash tools/unpack_coco.sh

# 生成清單
python tools/gen_lists.py --coco-path ./data/coco --output-dir ./data

echo "✓ COCO 資料集準備完成！"
```

## 注意事項

1. **網路傳輸**：大檔案建議使用 rsync 或分割傳輸
2. **儲存空間**：確保目標機器有足夠空間（至少 40GB）
3. **檔案完整性**：務必驗證 MD5 確保傳輸無誤
4. **.gitignore**：`data/coco/` 已被忽略，不會被 git 追蹤