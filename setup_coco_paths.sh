#!/bin/bash
# 設定 COCO 資料集路徑（避免重新下載）

# 建立 coco 目錄結構
mkdir -p yolov7/coco

# 生成資料清單（假設資料在 /data/coco）
echo "生成訓練集清單..."
find /data/coco/train2017 -type f -name "*.jpg" | sort > yolov7/coco/train2017.txt

echo "生成驗證集清單..."
find /data/coco/val2017 -type f -name "*.jpg" | sort > yolov7/coco/val2017.txt

# 如果需要測試集
if [ -d "/data/coco/test2017" ]; then
    echo "生成測試集清單..."
    find /data/coco/test2017 -type f -name "*.jpg" | sort > yolov7/coco/test-dev2017.txt
else
    # 建立空檔案避免錯誤
    touch yolov7/coco/test-dev2017.txt
fi

# 建立標籤目錄連結（如果存在）
if [ -d "/data/coco/labels" ]; then
    ln -sf /data/coco/labels yolov7/coco/labels
fi

echo "完成！檔案清單已生成："
wc -l yolov7/coco/*.txt