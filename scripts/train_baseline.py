#!/usr/bin/env python3
"""
YOLOv7-tiny Baseline 訓練包裝腳本
根據 Baseline Spec v1.0 的要求進行訓練
"""

import os
import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class BaselineTrainer:
    """
    Baseline 訓練管理器
    """
    
    def __init__(self, config):
        self.config = config
        self.yolov7_dir = Path("yolov7")
        self.exp_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def setup_environment(self):
        """設定訓練環境"""
        print("=== 設定訓練環境 ===")
        
        # 設定決定性訓練參數
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = '0'
        
        # 設定 Python 隨機種子
        import random
        import numpy as np
        import torch
        
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print("  ✓ 隨機種子設定完成")
        print("  ✓ CUDA 決定性模式啟用")
    
    def prepare_data_config(self):
        """準備資料配置檔案"""
        print("\n=== 準備資料配置 ===")
        
        # 建立自定義 COCO 配置
        coco_config = {
            'train': str(Path(self.config['data_path']) / 'train2017.txt'),
            'val': str(Path(self.config['data_path']) / 'val2017.txt'),
            'test': str(Path(self.config['data_path']) / 'val2017.txt'),
            'nc': 80,
            'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                     'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        }
        
        # 儲存配置檔案
        data_yaml_path = Path('data/coco_baseline.yaml')
        data_yaml_path.parent.mkdir(exist_ok=True)
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(coco_config, f, sort_keys=False)
        
        print(f"  ✓ 資料配置已儲存至: {data_yaml_path}")
        return str(data_yaml_path)
    
    def prepare_hyp_config(self):
        """準備超參數配置（包含 Mosaic/MixUp 控制）"""
        print("\n=== 準備超參數配置 ===")
        
        # 基於 hyp.scratch.tiny.yaml 但調整部分參數
        hyp = {
            'lr0': 0.01,  # 初始學習率
            'lrf': 0.01,  # 最終學習率因子
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,  # 將在訓練中動態控制
            'mixup': 0.05,  # 將在訓練中動態控制
            'copy_paste': 0.0,
            'paste_in': 0.0,
            'loss_ota': 1,
            'label_smoothing': 0.05  # 添加 label smoothing
        }
        
        # 儲存超參數配置
        hyp_yaml_path = Path('data/hyp_baseline.yaml')
        with open(hyp_yaml_path, 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        
        print(f"  ✓ 超參數配置已儲存至: {hyp_yaml_path}")
        return str(hyp_yaml_path)
    
    def modify_train_script(self):
        """修改訓練腳本以支援 Mosaic/MixUp 動態控制"""
        print("\n=== 準備訓練腳本修改 ===")
        
        # 複製原始訓練腳本
        original_train = self.yolov7_dir / "train.py"
        modified_train = self.yolov7_dir / "train_baseline.py"
        
        if not modified_train.exists():
            shutil.copy(original_train, modified_train)
            
            # 讀取並修改腳本
            with open(modified_train, 'r') as f:
                content = f.read()
            
            # 在訓練循環中添加 Mosaic/MixUp 控制邏輯
            mosaic_control = """
    # Baseline Spec: 關閉 Mosaic/MixUp 在 epoch 261-300
    if epoch >= 260:
        hyp['mosaic'] = 0.0
        hyp['mixup'] = 0.0
        if rank in [-1, 0]:
            logger.info('Epoch %d: Disabling Mosaic and MixUp for final epochs' % (epoch + 1))
"""
            
            # 尋找合適的插入點（在 epoch 循環開始處）
            insert_marker = "for epoch in range(start_epoch, epochs):"
            if insert_marker in content:
                parts = content.split(insert_marker)
                # 在 epoch 循環後插入控制邏輯
                content = parts[0] + insert_marker + "\n" + mosaic_control + parts[1]
            
            # 儲存修改後的腳本
            with open(modified_train, 'w') as f:
                f.write(content)
            
            print(f"  ✓ 訓練腳本已修改: {modified_train}")
        else:
            print(f"  ✓ 使用現有修改腳本: {modified_train}")
        
        return str(modified_train)
    
    def build_train_command(self, data_yaml, hyp_yaml, train_script):
        """建立訓練命令"""
        cmd = [
            "python", train_script,
            "--img", str(self.config['img_size']),
            "--batch", str(self.config['batch_size']),
            "--epochs", str(self.config['epochs']),
            "--data", data_yaml,
            "--cfg", "cfg/training/yolov7-tiny.yaml",
            "--weights", self.config['weights'],
            "--device", str(self.config['device']),
            "--workers", str(self.config['workers']),
            "--hyp", hyp_yaml,
            "--name", self.exp_name,
            "--project", "runs/train"
        ]
        
        # 添加其他參數
        if self.config.get('amp', True):
            # YOLOv7 可能使用不同的參數名稱
            pass  # AMP 通常是預設開啟的
        
        if self.config.get('save_period'):
            cmd.extend(["--save-period", str(self.config['save_period'])])
        
        return cmd
    
    def start_validation_watcher(self):
        """啟動背景驗證監控"""
        print("\n=== 啟動背景驗證監控 ===")
        
        watch_dir = self.yolov7_dir / f"runs/train/{self.exp_name}/weights"
        watcher_cmd = [
            "bash", "../tools/val_watcher.sh",
            str(watch_dir),
            "60",  # 檢查間隔
            str(self.config['img_size']),
            "0.001",  # conf_thres
            "0.65",  # iou_thres
            "data/coco_baseline.yaml"
        ]
        
        # 在背景執行
        watcher_process = subprocess.Popen(
            watcher_cmd,
            cwd=str(self.yolov7_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"  ✓ 驗證監控已啟動 (PID: {watcher_process.pid})")
        return watcher_process
    
    def train(self):
        """執行訓練"""
        print("\n=== 開始訓練 ===")
        
        # 設定環境
        self.setup_environment()
        
        # 準備配置檔案
        data_yaml = self.prepare_data_config()
        hyp_yaml = self.prepare_hyp_config()
        
        # 修改訓練腳本
        train_script = self.modify_train_script()
        
        # 建立訓練命令
        train_cmd = self.build_train_command(data_yaml, hyp_yaml, train_script)
        
        print(f"\n訓練命令:")
        print(" ".join(train_cmd))
        
        # 啟動背景驗證（如果需要）
        watcher_process = None
        if self.config.get('background_validation', True):
            watcher_process = self.start_validation_watcher()
        
        # 執行訓練
        print("\n開始訓練...")
        print("=" * 50)
        
        try:
            result = subprocess.run(
                train_cmd,
                cwd=str(self.yolov7_dir),
                check=True
            )
            print("\n✓ 訓練完成！")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ 訓練失敗: {e}")
            sys.exit(1)
        finally:
            # 停止背景驗證
            if watcher_process:
                watcher_process.terminate()
                print("  ✓ 背景驗證已停止")
        
        # 輸出結果路徑
        weights_dir = self.yolov7_dir / f"runs/train/{self.exp_name}/weights"
        print(f"\n訓練結果:")
        print(f"  權重目錄: {weights_dir}")
        print(f"  最佳權重: {weights_dir}/best.pt")
        print(f"  最後權重: {weights_dir}/last.pt")


def main():
    parser = argparse.ArgumentParser(description='YOLOv7-tiny Baseline 訓練')
    parser.add_argument('--img-size', type=int, default=320,
                        help='輸入影像尺寸')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                        help='訓練 epochs')
    parser.add_argument('--weights', type=str, default='yolov7-tiny.pt',
                        help='預訓練權重路徑')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA 裝置')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--data-path', type=str, default='.',
                        help='資料清單路徑')
    parser.add_argument('--save-period', type=int, default=25,
                        help='儲存 checkpoint 的週期')
    parser.add_argument('--no-amp', action='store_true',
                        help='停用 AMP 混合精度訓練')
    parser.add_argument('--no-validation', action='store_true',
                        help='停用背景驗證')
    
    args = parser.parse_args()
    
    # 建立配置
    config = {
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weights': args.weights,
        'device': args.device,
        'workers': args.workers,
        'data_path': args.data_path,
        'save_period': args.save_period,
        'amp': not args.no_amp,
        'background_validation': not args.no_validation
    }
    
    # 建立訓練器並執行
    trainer = BaselineTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()