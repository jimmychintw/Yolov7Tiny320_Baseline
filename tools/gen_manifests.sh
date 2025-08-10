#!/bin/bash
# 生成 SHA256 校驗檔案
# 根據 CLAUDE.md 中的指令

set -e

# 預設路徑
DATA_DIR="${DATA_DIR:-./data}"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 生成 SHA256 校驗檔案 ===${NC}"

# 檢查清單檔案是否存在
for file in train.txt val.txt calib.txt; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo -e "${RED}錯誤: 找不到 $DATA_DIR/$file${NC}"
        echo "請先執行: python tools/gen_lists.py --output-dir $DATA_DIR"
        exit 1
    fi
done

# 生成訓練集 SHA256
echo -e "${YELLOW}生成訓練集 SHA256 校驗檔...${NC}"
if [ -f "$DATA_DIR/train.txt" ]; then
    cat "$DATA_DIR/train.txt" | xargs -I{} shasum -a 256 "{}" > "$DATA_DIR/manifest_train.sha256"
    COUNT=$(wc -l < "$DATA_DIR/manifest_train.sha256")
    echo -e "${GREEN}  ✓ 已生成 $COUNT 個訓練集檔案的 SHA256${NC}"
fi

# 生成驗證集 SHA256
echo -e "${YELLOW}生成驗證集 SHA256 校驗檔...${NC}"
if [ -f "$DATA_DIR/val.txt" ]; then
    cat "$DATA_DIR/val.txt" | xargs -I{} shasum -a 256 "{}" > "$DATA_DIR/manifest_val.sha256"
    COUNT=$(wc -l < "$DATA_DIR/manifest_val.sha256")
    echo -e "${GREEN}  ✓ 已生成 $COUNT 個驗證集檔案的 SHA256${NC}"
fi

# 生成校正集 SHA256
echo -e "${YELLOW}生成校正集 SHA256 校驗檔...${NC}"
if [ -f "$DATA_DIR/calib.txt" ]; then
    cat "$DATA_DIR/calib.txt" | xargs -I{} shasum -a 256 "{}" > "$DATA_DIR/manifest_calib.sha256"
    COUNT=$(wc -l < "$DATA_DIR/manifest_calib.sha256")
    echo -e "${GREEN}  ✓ 已生成 $COUNT 個校正集檔案的 SHA256${NC}"
fi

echo -e "${GREEN}=== 完成！===${NC}"
echo "生成的檔案："
ls -lh "$DATA_DIR"/manifest_*.sha256