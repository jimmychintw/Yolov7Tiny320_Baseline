#!/usr/bin/env python3
"""
生成 COCO 2017 訓練/驗證/校正集清單
根據 Baseline Spec v1.0 的要求
"""

import os
import subprocess
from pathlib import Path
import argparse


def generate_file_lists(coco_path="/data/coco", output_dir="."):
    """
    生成訓練、驗證和校正集清單
    
    Args:
        coco_path: COCO 資料集路徑
        output_dir: 輸出清單檔案的目錄
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練集清單
    train_dir = Path(coco_path) / "train2017"
    val_dir = Path(coco_path) / "val2017"
    
    print(f"生成訓練集清單...")
    train_images = sorted(train_dir.glob("*.jpg"))
    train_txt = output_dir / "train.txt"
    with open(train_txt, 'w') as f:
        for img in train_images:
            f.write(str(img.absolute()) + '\n')
    print(f"  訓練集: {len(train_images)} 張影像 -> {train_txt}")
    
    # 驗證集清單
    print(f"生成驗證集清單...")
    val_images = sorted(val_dir.glob("*.jpg"))
    val_txt = output_dir / "val.txt"
    with open(val_txt, 'w') as f:
        for img in val_images:
            f.write(str(img.absolute()) + '\n')
    print(f"  驗證集: {len(val_images)} 張影像 -> {val_txt}")
    
    # 校正集清單 (從驗證集中穩定抽樣 512 張)
    print(f"生成校正集清單...")
    calib_txt = output_dir / "calib.txt"
    calib_images = val_images[::10][:512]  # 每10張取1張，最多512張
    with open(calib_txt, 'w') as f:
        for img in calib_images:
            f.write(str(img.absolute()) + '\n')
    print(f"  校正集: {len(calib_images)} 張影像 -> {calib_txt}")
    
    return train_txt, val_txt, calib_txt


def generate_sha256_manifests(train_txt, val_txt, calib_txt, output_dir="."):
    """
    生成 SHA256 校驗檔案
    
    Args:
        train_txt: 訓練集清單檔案
        val_txt: 驗證集清單檔案
        calib_txt: 校正集清單檔案
        output_dir: 輸出目錄
    """
    output_dir = Path(output_dir)
    
    print("\n生成 SHA256 校驗檔...")
    
    manifests = [
        (train_txt, output_dir / "manifest_train.sha256"),
        (val_txt, output_dir / "manifest_val.sha256"),
        (calib_txt, output_dir / "manifest_calib.sha256")
    ]
    
    for list_file, manifest_file in manifests:
        print(f"  處理 {list_file.name}...")
        with open(list_file, 'r') as f:
            image_paths = f.read().strip().split('\n')
        
        with open(manifest_file, 'w') as mf:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    result = subprocess.run(
                        ['sha256sum', img_path],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        mf.write(result.stdout)
        
        print(f"    -> {manifest_file}")


def create_yolo_format_lists(train_txt, val_txt, output_dir="."):
    """
    建立 YOLOv7 格式的資料清單 (包含標籤路徑)
    """
    output_dir = Path(output_dir)
    
    print("\n生成 YOLOv7 格式清單...")
    
    # 訓練集 YOLO 格式
    yolo_train = output_dir / "train2017.txt"
    with open(train_txt, 'r') as f_in, open(yolo_train, 'w') as f_out:
        for line in f_in:
            img_path = line.strip()
            # 將影像路徑轉換為標籤路徑
            label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
            if '/train2017/' in img_path:
                f_out.write(img_path + '\n')
    print(f"  YOLOv7 訓練集: {yolo_train}")
    
    # 驗證集 YOLO 格式
    yolo_val = output_dir / "val2017.txt"
    with open(val_txt, 'r') as f_in, open(yolo_val, 'w') as f_out:
        for line in f_in:
            img_path = line.strip()
            if '/val2017/' in img_path:
                f_out.write(img_path + '\n')
    print(f"  YOLOv7 驗證集: {yolo_val}")
    
    return yolo_train, yolo_val


def main():
    parser = argparse.ArgumentParser(description='生成 COCO 2017 資料集清單')
    parser.add_argument('--coco-path', type=str, default='/data/coco',
                        help='COCO 資料集路徑')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='輸出目錄')
    parser.add_argument('--skip-sha256', action='store_true',
                        help='跳過 SHA256 校驗檔生成')
    parser.add_argument('--yolo-format', action='store_true',
                        help='生成 YOLOv7 格式清單')
    
    args = parser.parse_args()
    
    print(f"COCO 資料集路徑: {args.coco_path}")
    print(f"輸出目錄: {args.output_dir}\n")
    
    # 生成基本清單
    train_txt, val_txt, calib_txt = generate_file_lists(
        args.coco_path, args.output_dir
    )
    
    # 生成 SHA256 校驗檔
    if not args.skip_sha256:
        generate_sha256_manifests(
            train_txt, val_txt, calib_txt, args.output_dir
        )
    
    # 生成 YOLOv7 格式清單
    if args.yolo_format:
        create_yolo_format_lists(train_txt, val_txt, args.output_dir)
    
    print("\n完成！")


if __name__ == "__main__":
    main()