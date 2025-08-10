#!/bin/bash
# COCO 資料集壓縮腳本
# 將 data/coco 壓縮成 coco.tar.gz 或 coco.zip

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 設定路徑
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
COCO_DIR="$DATA_DIR/coco"

# 檢查 COCO 目錄是否存在
if [ ! -d "$COCO_DIR" ]; then
    echo -e "${RED}錯誤：找不到 COCO 資料集目錄 $COCO_DIR${NC}"
    exit 1
fi

# 顯示資料集資訊
echo -e "${BLUE}=== COCO 資料集壓縮 ===${NC}"
echo "資料集路徑：$COCO_DIR"

# 計算資料集大小
SIZE=$(du -sh "$COCO_DIR" | cut -f1)
echo "資料集大小：$SIZE"

# 計算檔案數量
TRAIN_COUNT=$(find "$COCO_DIR/train2017" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
VAL_COUNT=$(find "$COCO_DIR/val2017" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
echo "訓練集：$TRAIN_COUNT 張圖片"
echo "驗證集：$VAL_COUNT 張圖片"

# 選擇壓縮格式
echo -e "\n${YELLOW}選擇壓縮格式：${NC}"
echo "1) tar.gz (Linux/Mac 推薦，壓縮率高)"
echo "2) zip (Windows 相容性好)"
echo "3) tar.xz (最高壓縮率，但較慢)"
read -p "請選擇 [1-3] (預設: 1): " choice
choice=${choice:-1}

# 設定輸出檔名和壓縮命令
case $choice in
    1)
        OUTPUT="$DATA_DIR/coco.tar.gz"
        echo -e "\n${GREEN}使用 tar.gz 格式壓縮...${NC}"
        cd "$DATA_DIR"
        tar -czf coco.tar.gz coco/ \
            --checkpoint=.1000 \
            --checkpoint-action=echo="已處理 %u 個檔案..." \
            --totals
        ;;
    2)
        OUTPUT="$DATA_DIR/coco.zip"
        echo -e "\n${GREEN}使用 zip 格式壓縮...${NC}"
        cd "$DATA_DIR"
        zip -r coco.zip coco/ -q
        ;;
    3)
        OUTPUT="$DATA_DIR/coco.tar.xz"
        echo -e "\n${GREEN}使用 tar.xz 格式壓縮（這需要較長時間）...${NC}"
        cd "$DATA_DIR"
        tar -cJf coco.tar.xz coco/ \
            --checkpoint=.1000 \
            --checkpoint-action=echo="已處理 %u 個檔案..."
        ;;
    *)
        echo -e "${RED}無效選擇${NC}"
        exit 1
        ;;
esac

# 顯示結果
if [ -f "$OUTPUT" ]; then
    COMPRESSED_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
    echo -e "\n${GREEN}✓ 壓縮完成！${NC}"
    echo "輸出檔案：$OUTPUT"
    echo "壓縮後大小：$COMPRESSED_SIZE"
    echo -e "\n${BLUE}傳輸指令：${NC}"
    echo "# 使用 scp 傳輸到遠端："
    echo "scp $OUTPUT user@remote:/path/to/destination/"
    echo ""
    echo "# 或使用 rsync（支援續傳）："
    echo "rsync -avP $OUTPUT user@remote:/path/to/destination/"
    
    # 生成 MD5 校驗碼
    echo -e "\n${YELLOW}生成 MD5 校驗碼...${NC}"
    md5sum "$OUTPUT" > "${OUTPUT}.md5"
    echo "校驗檔：${OUTPUT}.md5"
    cat "${OUTPUT}.md5"
else
    echo -e "${RED}✗ 壓縮失敗！${NC}"
    exit 1
fi