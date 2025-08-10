#!/bin/bash
#
# YOLOv7-tiny Baseline 環境設定腳本
# 根據 Baseline Spec v1.0 的要求
#

set -e  # 遇到錯誤時停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# 檢查系統需求
check_system() {
    log_step "檢查系統需求"
    
    # 檢查 Python 版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        log_info "Python 版本: $PYTHON_VERSION"
        
        # 檢查是否為 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_info "✓ Python 版本符合需求 (>=3.8)"
        else
            log_error "Python 版本過舊，需要 3.8 或更新版本"
            exit 1
        fi
    else
        log_error "找不到 python3 命令"
        exit 1
    fi
    
    # 檢查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log_info "CUDA 版本: $CUDA_VERSION"
        
        # 檢查 CUDA 版本是否符合需求 (12.1)
        if [[ "$CUDA_VERSION" == "12.1" ]]; then
            log_info "✓ CUDA 版本符合需求 (12.1)"
        else
            log_warn "CUDA 版本與 Baseline Spec 不符，可能影響結果重現性"
            log_warn "建議版本: 12.1，目前版本: $CUDA_VERSION"
        fi
    else
        log_warn "未檢測到 NVIDIA GPU 或 CUDA 驅動"
    fi
    
    # 檢查可用空間
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -gt 50 ]; then
        log_info "✓ 磁碟空間充足 (${AVAILABLE_SPACE}GB 可用)"
    else
        log_warn "磁碟空間不足，建議至少 50GB 可用空間"
    fi
}

# 設定決定性環境變數
setup_deterministic_env() {
    log_step "設定決定性環境變數"
    
    # 建立環境變數設定檔
    cat > .env << 'EOF'
# YOLOv7-tiny Baseline 環境變數
# 用於確保訓練結果的可重現性

# CUDA 決定性設定
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=0

# CUDA 快取設定
export CUDA_CACHE_DISABLE=1

# OpenMP 設定
export OMP_NUM_THREADS=1

# 其他決定性設定
export PYTHONDONTWRITEBYTECODE=1
EOF
    
    # 載入環境變數
    source .env
    
    log_info "✓ 決定性環境變數已設定"
    log_info "  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
    log_info "  PYTHONHASHSEED=$PYTHONHASHSEED"
}

# 建立虛擬環境
setup_venv() {
    log_step "建立 Python 虛擬環境"
    
    VENV_NAME="venv_baseline"
    
    if [ -d "$VENV_NAME" ]; then
        log_info "虛擬環境已存在: $VENV_NAME"
        read -p "是否重新建立？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            log_info "已刪除現有虛擬環境"
        else
            log_info "使用現有虛擬環境"
            return
        fi
    fi
    
    # 建立虛擬環境
    python3 -m venv "$VENV_NAME"
    log_info "✓ 虛擬環境建立完成: $VENV_NAME"
    
    # 啟用虛擬環境
    source "$VENV_NAME/bin/activate"
    
    # 升級 pip
    pip install --upgrade pip
    
    log_info "✓ 虛擬環境已啟用"
}

# 安裝 Python 套件
install_packages() {
    log_step "安裝 Python 套件"
    
    # 檢查 requirements.txt
    if [ ! -f "requirements.txt" ]; then
        log_error "找不到 requirements.txt"
        exit 1
    fi
    
    log_info "安裝基礎套件..."
    
    # 先安裝 PyTorch (指定版本和 CUDA)
    if command -v nvidia-smi &> /dev/null; then
        log_info "安裝 PyTorch with CUDA 12.1..."
        pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    else
        log_info "安裝 PyTorch CPU 版本..."
        pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安裝其他套件
    log_info "安裝其他套件..."
    pip install -r requirements.txt
    
    log_info "✓ 套件安裝完成"
    
    # 驗證關鍵套件版本
    log_info "驗證套件版本:"
    python3 -c "
import torch, torchvision, onnx, onnxruntime
print(f'  PyTorch: {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  ONNX: {onnx.__version__}')
print(f'  ONNXRuntime: {onnxruntime.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device count: {torch.cuda.device_count()}')
    print(f'  CUDA device name: {torch.cuda.get_device_name(0)}')
"
}

# 設定 YOLOv7 倉庫
setup_yolov7() {
    log_step "設定 YOLOv7 倉庫"
    
    if [ ! -d "yolov7" ]; then
        log_info "克隆 YOLOv7 倉庫..."
        git clone https://github.com/WongKinYiu/yolov7.git
        
        # 固定到指定 commit (可選)
        # cd yolov7 && git checkout <specific_commit> && cd ..
    else
        log_info "YOLOv7 倉庫已存在"
    fi
    
    # 下載預訓練權重
    WEIGHTS_PATH="yolov7/yolov7-tiny.pt"
    if [ ! -f "$WEIGHTS_PATH" ]; then
        log_info "下載 YOLOv7-tiny 預訓練權重..."
        cd yolov7
        wget -O yolov7-tiny.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
        cd ..
        
        if [ -f "$WEIGHTS_PATH" ]; then
            log_info "✓ 權重下載完成: $WEIGHTS_PATH"
        else
            log_error "權重下載失敗"
            exit 1
        fi
    else
        log_info "✓ 預訓練權重已存在: $WEIGHTS_PATH"
    fi
}

# 設定執行權限
setup_permissions() {
    log_step "設定執行權限"
    
    chmod +x tools/*.sh 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    chmod +x scripts/*.py 2>/dev/null || true
    chmod +x tools/*.py 2>/dev/null || true
    
    log_info "✓ 執行權限設定完成"
}

# 建立初始目錄結構
create_directories() {
    log_step "建立目錄結構"
    
    mkdir -p data
    mkdir -p runs
    mkdir -p val_results
    mkdir -p weights
    mkdir -p logs
    
    log_info "✓ 目錄結構建立完成"
}

# 生成啟動腳本
create_launcher() {
    log_step "生成啟動腳本"
    
    cat > activate_baseline.sh << 'EOF'
#!/bin/bash
# YOLOv7-tiny Baseline 啟動腳本
# 用法: source activate_baseline.sh

echo "啟動 YOLOv7-tiny Baseline 環境..."

# 啟用虛擬環境
if [ -d "venv_baseline" ]; then
    source venv_baseline/bin/activate
    echo "✓ 虛擬環境已啟用"
else
    echo "警告: 找不到虛擬環境 venv_baseline"
fi

# 載入環境變數
if [ -f ".env" ]; then
    source .env
    echo "✓ 環境變數已載入"
fi

# 設定 Python 路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/yolov7"

echo "環境設定完成！"
echo "可用命令:"
echo "  make help    - 查看所有可用目標"
echo "  make baseline - 執行完整 baseline 流程"
echo "  make status  - 檢查專案狀態"
EOF
    
    chmod +x activate_baseline.sh
    log_info "✓ 啟動腳本已建立: activate_baseline.sh"
}

# 生成環境記錄
generate_env_record() {
    log_step "生成環境記錄"
    
    cat > ENV_SETUP.md << EOF
# 環境設定記錄

設定時間: $(date '+%Y-%m-%d %H:%M:%S')
主機名稱: $(hostname)
使用者: $(whoami)

## 系統資訊
\`\`\`
$(uname -a)
\`\`\`

## Python 版本
\`\`\`
$(python3 --version)
\`\`\`

## CUDA 資訊
\`\`\`
$(nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected")
\`\`\`

## 磁碟空間
\`\`\`
$(df -h .)
\`\`\`

## 套件版本
\`\`\`
$(pip list | grep -E "(torch|onnx|numpy|opencv)" 2>/dev/null || echo "Virtual environment not activated")
\`\`\`

## YOLOv7 Commit
\`\`\`
$(cd yolov7 && git rev-parse HEAD 2>/dev/null || echo "Not a git repository")
\`\`\`

## 環境變數
\`\`\`
CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG
PYTHONHASHSEED=$PYTHONHASHSEED
\`\`\`
EOF
    
    log_info "✓ 環境記錄已生成: ENV_SETUP.md"
}

# 主函數
main() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "YOLOv7-tiny Baseline 環境設定腳本"
    echo "根據 Baseline Spec v1.0 建立環境"
    echo "========================================"
    echo -e "${NC}"
    
    # 檢查是否在正確的目錄
    if [ ! -f "CLAUDE.md" ]; then
        log_error "請在專案根目錄執行此腳本"
        exit 1
    fi
    
    # 執行設定步驟
    check_system
    setup_deterministic_env
    setup_venv
    install_packages
    setup_yolov7
    create_directories
    setup_permissions
    create_launcher
    generate_env_record
    
    echo -e "\n${GREEN}========================================"
    echo "環境設定完成！"
    echo "=======================================${NC}"
    echo ""
    echo "下一步:"
    echo "1. 執行: source activate_baseline.sh"
    echo "2. 執行: make help"
    echo "3. 執行: make baseline"
    echo ""
    echo "或者手動執行各個步驟:"
    echo "- make setup   # 環境設定"
    echo "- make data    # 資料準備"  
    echo "- make train   # 模型訓練"
    echo "- make export_onnx  # ONNX 導出"
    echo "- make ptq     # PTQ 量化"
    echo "- make eval    # 模型評測"
}

# 執行主函數
main "$@"