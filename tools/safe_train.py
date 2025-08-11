#!/usr/bin/env python3
"""
安全訓練腳本 - 確保不會自動下載資料集
使用方式：python tools/safe_train.py --epochs 300
"""

import os
import sys
import subprocess
from pathlib import Path

def check_prerequisites():
    """檢查所有前置條件"""
    project_root = Path(__file__).parent.parent
    errors = []
    
    # 1. 檢查資料集目錄
    data_dirs = [
        project_root / 'data' / 'coco' / 'train2017',
        project_root / 'data' / 'coco' / 'val2017',
    ]
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            errors.append(f"找不到資料目錄 {data_dir}")
        else:
            # 檢查是否有圖片
            images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
            if not images:
                errors.append(f"{data_dir} 目錄存在但沒有圖片")
            else:
                print(f"✓ {data_dir.name}: {len(images)} 張圖片")
    
    # 2. 檢查標註目錄
    anno_dir = project_root / 'data' / 'coco' / 'annotations'
    if not anno_dir.exists():
        print(f"⚠️  警告：找不到標註目錄 {anno_dir}")
        print("   如果使用 YOLO 格式標籤，請確認標籤在 labels 目錄")
    else:
        print(f"✓ 標註目錄: {anno_dir}")
    
    # 3. 檢查資料清單（選擇性）
    list_files = [
        project_root / 'train.txt',
        project_root / 'val.txt',
    ]
    
    has_lists = True
    for file_path in list_files:
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                print(f"✓ {file_path.name}: {len(lines)} 個檔案（清單）")
        else:
            has_lists = False
    
    if not has_lists:
        print("ℹ️  未找到資料清單檔案，將使用目錄模式")
    
    # 4. 檢查預訓練權重
    weights_path = project_root / 'yolov7' / 'yolov7-tiny.pt'
    if not weights_path.exists():
        errors.append(f"找不到預訓練權重 {weights_path}")
    else:
        print(f"✓ 預訓練權重: {weights_path}")
    
    # 5. 檢查 YOLOv7 目錄
    yolov7_dir = project_root / 'yolov7'
    if not yolov7_dir.exists():
        errors.append("找不到 yolov7 目錄")
    else:
        print(f"✓ YOLOv7 目錄: {yolov7_dir}")
    
    # 6. 檢查資料配置
    data_yaml = project_root / 'data' / 'coco_local.yaml'
    if not data_yaml.exists():
        errors.append(f"找不到資料配置 {data_yaml}")
    else:
        print(f"✓ 資料配置: {data_yaml}")
    
    # 如果有錯誤，顯示並退出
    if errors:
        print("\n❌ 發現以下問題：")
        for error in errors:
            print(f"  - {error}")
        print("\n請先解決這些問題再開始訓練")
        print("\n建議步驟：")
        print("1. 確保 COCO 資料集在專案的 data/coco/ 目錄：")
        print("   - data/coco/train2017/ (訓練圖片)")
        print("   - data/coco/val2017/ (驗證圖片)")
        print("   - data/coco/annotations/ (標註檔案)")
        print("2. 執行 python tools/gen_lists.py 生成清單（選擇性）")
        print("3. 下載 yolov7-tiny.pt 到 yolov7 目錄")
        sys.exit(1)
    
    print("\n✓ 所有檢查通過，可以開始訓練\n")
    return True

def main():
    """主程式"""
    import argparse
    parser = argparse.ArgumentParser(description='安全訓練腳本')
    parser.add_argument('--epochs', type=int, default=300, help='訓練 epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--img', type=int, default=320, help='圖片大小')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--name', default='safe_train', help='訓練名稱')
    
    args = parser.parse_args()
    
    # 檢查前置條件
    check_prerequisites()
    
    # 準備路徑
    project_root = Path(__file__).parent.parent
    yolov7_dir = project_root / 'yolov7'
    data_yaml = project_root / 'data' / 'coco_local.yaml'
    
    # 設定環境變數，禁用自動下載
    os.environ['WANDB_MODE'] = 'offline'  # 避免 wandb 連線
    os.environ['DATASET_AUTODOWNLOAD'] = 'false'  # 禁用自動下載（如果支援）
    
    # 構建訓練命令
    train_cmd = [
        'python', 'train.py',
        '--img', str(args.img),
        '--batch', str(args.batch),
        '--epochs', str(args.epochs),
        '--cfg', 'cfg/training/yolov7-tiny.yaml',
        '--data', str(data_yaml.absolute()),
        '--weights', 'yolov7-tiny.pt',
        '--device', args.device,
        '--workers', str(args.workers),
        '--hyp', 'data/hyp.scratch.tiny.yaml',
        '--name', args.name,
        '--exist-ok',  # 允許覆蓋已存在的目錄
    ]
    
    print("執行訓練命令：")
    print(' '.join(train_cmd))
    print()
    
    # 切換到 yolov7 目錄
    os.chdir(yolov7_dir)
    
    try:
        # 執行訓練
        result = subprocess.run(train_cmd, check=False)
        
        # 檢查是否因為找不到資料而失敗
        if result.returncode != 0:
            print("\n⚠️  訓練可能失敗了")
            print("如果是因為找不到資料集，請：")
            print("1. 確認資料路徑正確")
            print("2. 使用 python tools/gen_lists.py 重新生成清單")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n訓練被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print(f"\n錯誤：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()