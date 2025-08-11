#!/usr/bin/env python3
"""
自適應 Batch Size 和學習率的訓練腳本
根據 GPU VRAM 自動調整參數
"""

import torch
import subprocess
import yaml
import os
import sys
from pathlib import Path
import json

# GPU 配置表 (VRAM GB -> 建議 batch size)
GPU_CONFIGS = {
    # 消費級 GPU
    16: {'batch_size': 64, 'accumulate': 2},    # 4060 Ti, 4070 Ti
    24: {'batch_size': 128, 'accumulate': 1},   # 3090, 4090
    32: {'batch_size': 256, 'accumulate': 1},   # 5090 (預估)
    40: {'batch_size': 320, 'accumulate': 1},   # A100 40GB
    48: {'batch_size': 384, 'accumulate': 1},   # A6000, L40
    80: {'batch_size': 640, 'accumulate': 1},   # A100 80GB, H100
    192: {'batch_size': 1536, 'accumulate': 1}, # B200
}

# 基準配置 (YOLOv7-tiny 官方)
BASE_CONFIG = {
    'batch_size': 128,
    'lr0': 0.01,  # 初始學習率
    'warmup_epochs': 3.0,
    'warmup_bias_lr': 0.1,
}

def get_gpu_memory():
    """獲取 GPU VRAM (GB)"""
    if not torch.cuda.is_available():
        print("警告：未檢測到 CUDA，使用 CPU 模式")
        return 0
    
    try:
        # 獲取第一個 GPU 的記憶體
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        gpu_name = props.name
        print(f"檢測到 GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        return vram_gb
    except Exception as e:
        print(f"獲取 GPU 資訊失敗: {e}")
        return 0

def get_optimal_batch_size(vram_gb, img_size=320):
    """根據 VRAM 推薦 batch size"""
    # 根據圖片大小調整
    size_factor = (320 / img_size) ** 2
    
    # 找最接近的配置
    best_config = GPU_CONFIGS[24]  # 預設 4090 配置
    for vram_threshold in sorted(GPU_CONFIGS.keys()):
        if vram_gb >= vram_threshold * 0.9:  # 留 10% 餘量
            best_config = GPU_CONFIGS[vram_threshold]
    
    # 根據圖片大小調整
    adjusted_bs = int(best_config['batch_size'] * size_factor)
    
    # 確保是 8 的倍數（效能優化）
    adjusted_bs = (adjusted_bs // 8) * 8
    adjusted_bs = max(8, adjusted_bs)  # 最小 8
    
    return adjusted_bs, best_config['accumulate']

def calculate_learning_rate(batch_size, accumulate=1):
    """根據 batch size 計算學習率 (Linear Scaling Rule)"""
    effective_batch = batch_size * accumulate
    base_batch = BASE_CONFIG['batch_size']
    base_lr = BASE_CONFIG['lr0']
    
    # Linear scaling
    scaled_lr = base_lr * (effective_batch / base_batch)
    
    # 限制最大學習率（防止不穩定）
    max_lr = base_lr * 8
    scaled_lr = min(scaled_lr, max_lr)
    
    # Warmup 也要調整
    warmup_bias_lr = BASE_CONFIG['warmup_bias_lr'] * (effective_batch / base_batch)
    
    return scaled_lr, warmup_bias_lr

def validate_dataset(data_yaml_path, yolov7_dir):
    """驗證資料集是否存在，不存在則報錯退出"""
    print(f"\n檢查資料集配置: {data_yaml_path}")
    
    # 讀取資料配置
    yaml_full_path = yolov7_dir / data_yaml_path if not Path(data_yaml_path).is_absolute() else Path(data_yaml_path)
    
    if not yaml_full_path.exists():
        print(f"❌ 錯誤：找不到資料配置檔 {yaml_full_path}")
        print("請確認檔案路徑是否正確")
        sys.exit(1)
    
    with open(yaml_full_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 專案根目錄（yolov7 的上層目錄）
    project_root = yolov7_dir.parent
    
    # 檢查必要的資料路徑
    required_keys = ['train', 'val']
    missing_data = []
    
    for key in required_keys:
        if key not in data:
            print(f"❌ 錯誤：資料配置缺少 '{key}' 欄位")
            sys.exit(1)
        
        data_path = data[key]
        
        # 處理相對路徑
        if not Path(data_path).is_absolute():
            # 相對於 yolov7 目錄（訓練時的工作目錄）
            full_path = yolov7_dir / data_path
            # 如果不存在，也檢查專案根目錄
            if not full_path.exists() and data_path.startswith('../'):
                # 處理 ../data/coco 這種相對路徑
                alt_path = project_root / data_path.replace('../', '')
                if alt_path.exists():
                    full_path = alt_path
        else:
            full_path = Path(data_path)
        
        # 檢查路徑是否存在
        if not full_path.exists():
            missing_data.append(f"  - {key}: {full_path}")
        else:
            # 檢查是目錄還是檔案
            if full_path.is_dir():
                # 檢查目錄是否有圖片
                images = list(full_path.glob('*.jpg')) + list(full_path.glob('*.png'))
                if not images:
                    missing_data.append(f"  - {key}: {full_path} (目錄存在但沒有圖片)")
                else:
                    print(f"  ✓ {key}: {full_path} ({len(images)} 張圖片)")
            elif full_path.is_file():
                # 檢查清單檔案
                with open(full_path, 'r') as f:
                    lines = f.readlines()
                    valid_lines = [l.strip() for l in lines if l.strip()]
                    if not valid_lines:
                        missing_data.append(f"  - {key}: {full_path} (檔案存在但沒有內容)")
                    else:
                        # 檢查第一個檔案是否存在
                        first_file = Path(valid_lines[0])
                        if not first_file.exists():
                            missing_data.append(f"  - {key}: 清單中的檔案不存在 ({first_file})")
                        else:
                            print(f"  ✓ {key}: {full_path} ({len(valid_lines)} 個檔案)")
    
    # 檢查標籤目錄（如果指定了）
    if 'train' in data:
        train_path = data['train']
        if not Path(train_path).is_absolute():
            train_full = yolov7_dir / train_path
        else:
            train_full = Path(train_path)
        
        # 推測標籤路徑
        if train_full.exists():
            if 'images' in str(train_full):
                label_path = Path(str(train_full).replace('images', 'labels'))
                if not label_path.exists():
                    print(f"  ⚠️  警告：找不到標籤目錄 {label_path}")
                    print(f"     請確認標籤檔案位置")
    
    # 如果有缺失的資料，報錯退出
    if missing_data:
        print("\n❌ 錯誤：找不到以下資料集路徑：")
        for item in missing_data:
            print(item)
        print("\n請先準備好資料集，確保：")
        print("1. COCO 資料集已下載到專案的 data/coco/ 目錄")
        print("   - data/coco/train2017/ (訓練圖片)")
        print("   - data/coco/val2017/ (驗證圖片)")
        print("   - data/coco/annotations/ (標註檔案)")
        print("2. 如果使用清單檔案，執行 python tools/gen_lists.py 生成")
        print("3. 確認 data/coco_local.yaml 中的路徑配置正確")
        print("\n建議執行：")
        print("  python tools/gen_lists.py  # 生成資料清單")
        print("  或確認資料集在 data/coco/ 目錄")
        sys.exit(1)
    
    print("✓ 資料集檢查通過\n")
    return True

def create_custom_hyp(base_hyp_path, output_path, lr0, warmup_bias_lr):
    """創建自定義超參數檔案"""
    with open(base_hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # 更新學習率相關參數
    hyp['lr0'] = lr0
    hyp['warmup_bias_lr'] = warmup_bias_lr
    
    # 寫入新檔案
    with open(output_path, 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='自適應 Batch Size 訓練')
    parser.add_argument('--img', type=int, default=320, help='輸入圖片大小')
    parser.add_argument('--epochs', type=int, default=300, help='訓練 epochs')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--name', default='adaptive', help='訓練名稱')
    parser.add_argument('--weights', default='yolov7-tiny.pt', help='預訓練權重')
    parser.add_argument('--data', default='data/coco.yaml', help='資料集配置檔')
    
    # 手動指定
    parser.add_argument('--batch-size', type=int, help='手動指定 batch size')
    parser.add_argument('--lr', type=float, help='手動指定學習率')
    parser.add_argument('--accumulate', type=int, help='梯度累積步數')
    
    args = parser.parse_args()
    
    # 設定 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    # 獲取 GPU 資訊
    vram_gb = get_gpu_memory()
    
    # 決定 batch size
    if args.batch_size:
        batch_size = args.batch_size
        accumulate = args.accumulate or 1
        print(f"使用手動指定 batch size: {batch_size}")
    else:
        batch_size, accumulate = get_optimal_batch_size(vram_gb, args.img)
        print(f"自動選擇 batch size: {batch_size} (accumulate: {accumulate})")
    
    # 計算學習率
    if args.lr:
        lr0 = args.lr
        warmup_bias_lr = lr0 * 0.1
        print(f"使用手動指定學習率: {lr0}")
    else:
        lr0, warmup_bias_lr = calculate_learning_rate(batch_size, accumulate)
        print(f"自動計算學習率: {lr0:.4f} (base: {BASE_CONFIG['lr0']})")
    
    # 有效 batch size
    effective_batch = batch_size * accumulate
    print(f"有效 batch size: {effective_batch}")
    
    # 準備路徑
    project_root = Path(__file__).parent.parent
    yolov7_dir = project_root / 'yolov7'
    
    # 驗證資料集存在（在訓練前檢查）
    validate_dataset(args.data, yolov7_dir)
    
    # 檢查預訓練權重
    weights_path = yolov7_dir / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    if not weights_path.exists():
        print(f"❌ 錯誤：找不到預訓練權重 {weights_path}")
        print("請下載 yolov7-tiny.pt 到 yolov7 目錄")
        print("下載連結：https://github.com/WongKinYiu/yolov7/releases")
        sys.exit(1)
    print(f"✓ 預訓練權重: {weights_path}")
    
    # 創建自定義超參數檔案
    base_hyp = yolov7_dir / 'data' / 'hyp.scratch.tiny.yaml'
    custom_hyp = project_root / 'data' / f'hyp_bs{effective_batch}.yaml'
    custom_hyp.parent.mkdir(exist_ok=True)
    
    create_custom_hyp(base_hyp, custom_hyp, lr0, warmup_bias_lr)
    print(f"創建自定義超參數: {custom_hyp}")
    
    # 構建訓練命令
    train_cmd = [
        'python', 'train.py',
        '--img', str(args.img),
        '--batch', str(batch_size),
        '--epochs', str(args.epochs),
        '--cfg', 'cfg/training/yolov7-tiny.yaml',
        '--data', args.data,
        '--weights', args.weights,
        '--device', args.device,
        '--workers', str(args.workers),
        '--hyp', str(custom_hyp.absolute()),
        '--name', f'{args.name}_bs{effective_batch}_lr{lr0:.4f}',
    ]
    
    print("\n" + "="*60)
    print("訓練配置摘要：")
    print(f"  GPU: {vram_gb:.1f} GB VRAM")
    print(f"  Batch Size: {batch_size} × {accumulate} = {effective_batch}")
    print(f"  Learning Rate: {lr0:.4f}")
    print(f"  Image Size: {args.img}")
    print(f"  Epochs: {args.epochs}")
    print("="*60 + "\n")
    
    # 切換到 yolov7 目錄執行
    os.chdir(yolov7_dir)
    print(f"工作目錄: {os.getcwd()}")
    print(f"執行命令: {' '.join(train_cmd)}\n")
    
    # 執行訓練
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"訓練失敗: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n訓練被使用者中斷")
        sys.exit(0)

if __name__ == "__main__":
    main()