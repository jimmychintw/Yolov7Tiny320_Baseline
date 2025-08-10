#!/bin/bash
#
# 背景驗證監控腳本
# 根據 Baseline Spec v1.0 的要求
# 每隔指定時間檢查是否有新的 checkpoint，並自動執行驗證
#

# 設定參數
WATCH_DIR="${1:-./yolov7/runs/train/exp/weights}"
CHECK_INTERVAL="${2:-60}"  # 預設 60 秒
IMG_SIZE="${3:-320}"
CONF_THRES="${4:-0.001}"
IOU_THRES="${5:-0.65}"
DATA_YAML="${6:-./yolov7/data/coco.yaml}"
LOG_FILE="val_watcher.log"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 顯示使用說明
usage() {
    echo "使用方式: $0 [WATCH_DIR] [CHECK_INTERVAL] [IMG_SIZE] [CONF_THRES] [IOU_THRES] [DATA_YAML]"
    echo ""
    echo "參數:"
    echo "  WATCH_DIR      - 監控的權重目錄 (預設: ./yolov7/runs/train/exp/weights)"
    echo "  CHECK_INTERVAL - 檢查間隔秒數 (預設: 60)"
    echo "  IMG_SIZE       - 影像尺寸 (預設: 320)"
    echo "  CONF_THRES     - 信心度閾值 (預設: 0.001)"
    echo "  IOU_THRES      - IoU 閾值 (預設: 0.65)"
    echo "  DATA_YAML      - 資料配置檔案 (預設: ./yolov7/data/coco.yaml)"
    echo ""
    echo "範例:"
    echo "  $0"
    echo "  $0 ./runs/train/exp1/weights 30"
}

# 檢查參數
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

# 記錄訊息
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}"
            ;;
    esac
    
    echo "[${level}] ${timestamp} - ${message}" >> ${LOG_FILE}
}

# 執行驗證
run_validation() {
    local weight_file=$1
    local epoch=$(basename ${weight_file} .pt | sed 's/[^0-9]*//g')
    
    log_message INFO "開始驗證 ${weight_file}"
    
    # 建立驗證輸出目錄
    val_output_dir="./val_results/epoch_${epoch}"
    mkdir -p ${val_output_dir}
    
    # 執行驗證
    cd yolov7
    python test.py \
        --weights ${weight_file} \
        --data ${DATA_YAML} \
        --img ${IMG_SIZE} \
        --conf-thres ${CONF_THRES} \
        --iou-thres ${IOU_THRES} \
        --batch 1 \
        --device 0 \
        --save-txt \
        --save-conf \
        --project ../val_results \
        --name epoch_${epoch} \
        --exist-ok 2>&1 | tee ../val_results/epoch_${epoch}.log
    
    cd ..
    
    # 檢查驗證是否成功
    if [ $? -eq 0 ]; then
        log_message INFO "驗證完成: epoch ${epoch}"
        
        # 提取 mAP 結果
        mAP=$(grep "Average Precision" val_results/epoch_${epoch}.log | head -1 | awk '{print $NF}')
        if [ ! -z "$mAP" ]; then
            log_message INFO "Epoch ${epoch} mAP@0.5:0.95 = ${mAP}"
            echo "${epoch},${mAP}" >> val_results/mAP_history.csv
        fi
    else
        log_message ERROR "驗證失敗: epoch ${epoch}"
    fi
}

# 主監控循環
main() {
    log_message INFO "=== 背景驗證監控啟動 ==="
    log_message INFO "監控目錄: ${WATCH_DIR}"
    log_message INFO "檢查間隔: ${CHECK_INTERVAL} 秒"
    log_message INFO "影像尺寸: ${IMG_SIZE}"
    log_message INFO "信心度閾值: ${CONF_THRES}"
    log_message INFO "IoU 閾值: ${IOU_THRES}"
    log_message INFO "資料配置: ${DATA_YAML}"
    
    # 初始化已處理清單
    PROCESSED_FILE=".val_watcher_processed"
    touch ${PROCESSED_FILE}
    
    # 初始化 mAP 記錄檔
    if [ ! -f "val_results/mAP_history.csv" ]; then
        mkdir -p val_results
        echo "epoch,mAP@0.5:0.95" > val_results/mAP_history.csv
    fi
    
    # 監控循環
    while true; do
        # 檢查監控目錄是否存在
        if [ ! -d "${WATCH_DIR}" ]; then
            log_message WARN "監控目錄不存在: ${WATCH_DIR}"
            sleep ${CHECK_INTERVAL}
            continue
        fi
        
        # 尋找新的 checkpoint
        for weight_file in ${WATCH_DIR}/last*.pt ${WATCH_DIR}/best*.pt; do
            if [ -f "${weight_file}" ]; then
                # 檢查是否已處理過
                weight_basename=$(basename ${weight_file})
                weight_mtime=$(stat -f %m ${weight_file} 2>/dev/null || stat -c %Y ${weight_file} 2>/dev/null)
                weight_id="${weight_basename}_${weight_mtime}"
                
                if ! grep -q "^${weight_id}$" ${PROCESSED_FILE}; then
                    log_message INFO "發現新的 checkpoint: ${weight_file}"
                    
                    # 執行驗證
                    run_validation ${weight_file}
                    
                    # 記錄為已處理
                    echo "${weight_id}" >> ${PROCESSED_FILE}
                fi
            fi
        done
        
        # 檢查是否有特定 epoch 的 checkpoint (例如每 25 epoch)
        for epoch in 25 50 75 100 125 150 175 200 225 250 275 300; do
            epoch_file="${WATCH_DIR}/epoch${epoch}.pt"
            if [ -f "${epoch_file}" ]; then
                weight_mtime=$(stat -f %m ${epoch_file} 2>/dev/null || stat -c %Y ${epoch_file} 2>/dev/null)
                weight_id="epoch${epoch}.pt_${weight_mtime}"
                
                if ! grep -q "^${weight_id}$" ${PROCESSED_FILE}; then
                    log_message INFO "發現 epoch ${epoch} checkpoint"
                    run_validation ${epoch_file}
                    echo "${weight_id}" >> ${PROCESSED_FILE}
                fi
            fi
        done
        
        # 等待下一次檢查
        sleep ${CHECK_INTERVAL}
    done
}

# 捕捉中斷信號
trap 'log_message INFO "收到中斷信號，正在結束..."; exit 0' INT TERM

# 執行主程式
main