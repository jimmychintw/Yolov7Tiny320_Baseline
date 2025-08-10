#!/bin/bash
# COCO 資料集解壓縮腳本
# 自動檢測並解壓縮 coco.tar.gz, coco.zip 或 coco.tar.xz

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

echo -e "${BLUE}=== COCO 資料集解壓縮 ===${NC}"
echo "目標目錄：$DATA_DIR"

# 檢查是否已存在 COCO 目錄
if [ -d "$DATA_DIR/coco" ]; then
    echo -e "${YELLOW}警告：$DATA_DIR/coco 目錄已存在${NC}"
    read -p "是否覆蓋？[y/N]: " confirm
    if [[ $confirm != [yY] ]]; then
        echo "取消解壓縮"
        exit 0
    fi
    echo "移除舊目錄..."
    rm -rf "$DATA_DIR/coco"
fi

# 自動檢測壓縮檔
ARCHIVE=""
if [ -f "$DATA_DIR/coco.tar.gz" ]; then
    ARCHIVE="$DATA_DIR/coco.tar.gz"
    FORMAT="tar.gz"
elif [ -f "$DATA_DIR/coco.zip" ]; then
    ARCHIVE="$DATA_DIR/coco.zip"
    FORMAT="zip"
elif [ -f "$DATA_DIR/coco.tar.xz" ]; then
    ARCHIVE="$DATA_DIR/coco.tar.xz"
    FORMAT="tar.xz"
elif [ -f "$1" ]; then
    # 支援指定檔案路徑
    ARCHIVE="$1"
    # 自動檢測格式
    if [[ "$ARCHIVE" == *.tar.gz ]]; then
        FORMAT="tar.gz"
    elif [[ "$ARCHIVE" == *.zip ]]; then
        FORMAT="zip"
    elif [[ "$ARCHIVE" == *.tar.xz ]]; then
        FORMAT="tar.xz"
    else
        echo -e "${RED}錯誤：不支援的檔案格式${NC}"
        exit 1
    fi
else
    echo -e "${RED}錯誤：找不到 COCO 壓縮檔${NC}"
    echo "請確保以下檔案之一存在："
    echo "  - $DATA_DIR/coco.tar.gz"
    echo "  - $DATA_DIR/coco.zip"
    echo "  - $DATA_DIR/coco.tar.xz"
    echo "或指定檔案路徑："
    echo "  $0 /path/to/coco.tar.gz"
    exit 1
fi

echo "找到壓縮檔：$ARCHIVE"
echo "格式：$FORMAT"

# 檢查 MD5（如果存在）
if [ -f "${ARCHIVE}.md5" ]; then
    echo -e "${YELLOW}驗證 MD5 校驗碼...${NC}"
    cd "$(dirname "$ARCHIVE")"
    if md5sum -c "$(basename "${ARCHIVE}.md5")" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ MD5 校驗通過${NC}"
    else
        echo -e "${RED}✗ MD5 校驗失敗！檔案可能損壞${NC}"
        read -p "是否繼續？[y/N]: " confirm
        if [[ $confirm != [yY] ]]; then
            exit 1
        fi
    fi
fi

# 解壓縮
echo -e "\n${GREEN}開始解壓縮...${NC}"
cd "$DATA_DIR"

case $FORMAT in
    tar.gz)
        tar -xzf "$ARCHIVE" --checkpoint=.1000 --checkpoint-action=echo="已解壓 %u 個檔案..."
        ;;
    zip)
        unzip -q "$ARCHIVE"
        ;;
    tar.xz)
        tar -xJf "$ARCHIVE" --checkpoint=.1000 --checkpoint-action=echo="已解壓 %u 個檔案..."
        ;;
esac

# 驗證解壓縮結果
if [ -d "$DATA_DIR/coco" ]; then
    echo -e "\n${GREEN}✓ 解壓縮完成！${NC}"
    
    # 顯示資料集資訊
    TRAIN_COUNT=$(find "$DATA_DIR/coco/train2017" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
    VAL_COUNT=$(find "$DATA_DIR/coco/val2017" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
    SIZE=$(du -sh "$DATA_DIR/coco" | cut -f1)
    
    echo -e "\n${BLUE}資料集資訊：${NC}"
    echo "路徑：$DATA_DIR/coco"
    echo "大小：$SIZE"
    echo "訓練集：$TRAIN_COUNT 張圖片"
    echo "驗證集：$VAL_COUNT 張圖片"
    
    # 提示下一步
    echo -e "\n${YELLOW}下一步：${NC}"
    echo "1. 生成資料清單："
    echo "   python tools/gen_lists.py --coco-path $DATA_DIR/coco --output-dir $DATA_DIR"
    echo ""
    echo "2. 開始訓練："
    echo "   python tools/adaptive_train.py --epochs 300"
    
    # 詢問是否刪除壓縮檔
    echo -e "\n${YELLOW}壓縮檔處理：${NC}"
    read -p "是否刪除壓縮檔以節省空間？[y/N]: " delete_archive
    if [[ $delete_archive == [yY] ]]; then
        rm -f "$ARCHIVE" "${ARCHIVE}.md5"
        echo -e "${GREEN}✓ 已刪除壓縮檔${NC}"
    fi
else
    echo -e "${RED}✗ 解壓縮失敗！${NC}"
    exit 1
fi