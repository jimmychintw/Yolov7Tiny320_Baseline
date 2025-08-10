#!/usr/bin/env python3
"""
ONNX Runtime PTQ (Post-Training Quantization) 量化腳本
根據 Baseline Spec v1.0 的要求實作
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)
from onnxruntime.quantization.calibrate import CalibrationMethod


class COCOCalibrationDataReader(CalibrationDataReader):
    """
    COCO 資料集的校正資料讀取器
    """
    
    def __init__(self, calib_list_path, img_size=320, batch_size=1):
        """
        Args:
            calib_list_path: 校正集清單檔案路徑
            img_size: 輸入影像尺寸
            batch_size: 批次大小
        """
        self.img_size = img_size
        self.batch_size = batch_size
        
        # 讀取校正影像清單
        with open(calib_list_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        self.datasize = len(self.image_paths)
        self.enum_data_dicts = []
        
        # 預處理所有校正影像
        print(f"載入 {self.datasize} 張校正影像...")
        for img_path in self.image_paths:
            img_data = self.preprocess_image(img_path)
            self.enum_data_dicts.append({'images': img_data})
        
        self.iter = 0
    
    def preprocess_image(self, img_path):
        """
        影像預處理：letterbox + normalize
        符合 Baseline Spec 的要求
        """
        # 讀取影像
        img = Image.open(img_path).convert('RGB')
        
        # Letterbox 處理
        img = self.letterbox(img, new_shape=(self.img_size, self.img_size))
        
        # 轉換為 numpy array
        img = np.array(img, dtype=np.float32)
        
        # BGR 轉 RGB (如果需要)
        # img = img[:, :, ::-1]
        
        # 正規化 (0-255 -> 0-1)
        img = img / 255.0
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 增加 batch 維度
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img
    
    def letterbox(self, img, new_shape=(320, 320), color=(114, 114, 114)):
        """
        YOLOv7 letterbox 實作
        auto=False, scaleFill=False, scaleup=True
        """
        shape = img.size[::-1]  # PIL size is (width, height)
        
        # 計算縮放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 計算新尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # 分配 padding
        dw, dh = dw // 2, dh // 2
        
        # 調整影像大小
        if shape[::-1] != new_unpad:
            img = img.resize(new_unpad, Image.BILINEAR)
        
        # 添加 padding
        top, bottom = dh, dh + (new_shape[0] - new_unpad[1]) % 2
        left, right = dw, dw + (new_shape[1] - new_unpad[0]) % 2
        
        img_padded = Image.new('RGB', new_shape[::-1], color)
        img_padded.paste(img, (left, top))
        
        return img_padded
    
    def get_next(self):
        """
        獲取下一批次的校正資料
        """
        if self.iter >= self.datasize:
            return None
        
        data_dict = self.enum_data_dicts[self.iter]
        self.iter += 1
        
        return data_dict
    
    def rewind(self):
        """
        重置迭代器
        """
        self.iter = 0


def quantize_onnx_model(
    model_path,
    output_path,
    calib_list_path,
    method='percentile',
    percentile=99.99,
    img_size=320
):
    """
    執行 ONNX 模型的 PTQ 量化
    
    Args:
        model_path: 輸入 ONNX 模型路徑
        output_path: 輸出量化模型路徑
        calib_list_path: 校正集清單檔案
        method: 校正方法 ('percentile' 或 'minmax')
        percentile: percentile 方法的百分位數
        img_size: 輸入影像尺寸
    """
    print(f"\n=== ONNX Runtime PTQ 量化 ===")
    print(f"輸入模型: {model_path}")
    print(f"輸出模型: {output_path}")
    print(f"校正清單: {calib_list_path}")
    print(f"校正方法: {method} (percentile={percentile})")
    print(f"影像尺寸: {img_size}x{img_size}")
    
    # 建立校正資料讀取器
    calibration_reader = COCOCalibrationDataReader(
        calib_list_path,
        img_size=img_size
    )
    
    # 設定量化參數
    if method == 'percentile':
        calibrate_method = CalibrationMethod.Percentile
        extra_options = {'CalibPercentile': percentile}
    else:  # minmax
        calibrate_method = CalibrationMethod.MinMax
        extra_options = {}
    
    # 執行量化
    print("\n開始量化...")
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # 使用 QDQ 格式
        activation_type=QuantType.QUInt8,  # Activations: UINT8 (asymmetric)
        weight_type=QuantType.QInt8,  # Weights: INT8 (symmetric)
        per_channel=True,  # Weights per-channel
        reduce_range=False,
        calibrate_method=calibrate_method,
        extra_options=extra_options
    )
    
    print(f"\n量化完成！輸出至: {output_path}")
    
    # 驗證量化模型
    verify_quantized_model(output_path, img_size)
    
    # 生成 MD5 校驗碼
    generate_md5(output_path)


def verify_quantized_model(model_path, img_size=320):
    """
    驗證量化模型是否可正常載入和推論
    """
    print("\n驗證量化模型...")
    
    try:
        # 載入模型
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(model_path, sess_options)
        
        # 獲取輸入輸出資訊
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_names = [o.name for o in sess.get_outputs()]
        
        print(f"  輸入: {input_name} {input_shape}")
        print(f"  輸出: {output_names}")
        
        # 測試推論
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        outputs = sess.run(output_names, {input_name: dummy_input})
        
        print(f"  推論測試成功！輸出形狀: {[o.shape for o in outputs]}")
        
    except Exception as e:
        print(f"  錯誤: {e}")
        sys.exit(1)


def generate_md5(file_path):
    """
    生成檔案的 MD5 校驗碼
    """
    import hashlib
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    
    md5_file = file_path.replace('.onnx', '_md5.txt')
    with open(md5_file, 'w') as f:
        f.write(f"{md5_hash.hexdigest()}  {Path(file_path).name}\n")
    
    print(f"  MD5: {md5_hash.hexdigest()} -> {md5_file}")


def main():
    parser = argparse.ArgumentParser(description='ONNX Runtime PTQ 量化')
    parser.add_argument('--model', type=str, required=True,
                        help='輸入 ONNX 模型路徑')
    parser.add_argument('--calib', type=str, required=True,
                        help='校正集清單檔案路徑')
    parser.add_argument('--out', type=str, required=True,
                        help='輸出量化模型路徑')
    parser.add_argument('--method', type=str, default='percentile',
                        choices=['percentile', 'minmax'],
                        help='校正方法')
    parser.add_argument('--percentile', type=float, default=99.99,
                        help='Percentile 值 (僅用於 percentile 方法)')
    parser.add_argument('--img-size', type=int, default=320,
                        help='輸入影像尺寸')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案
    if not os.path.exists(args.model):
        print(f"錯誤: 找不到模型檔案 {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.calib):
        print(f"錯誤: 找不到校正清單 {args.calib}")
        sys.exit(1)
    
    # 執行量化
    quantize_onnx_model(
        args.model,
        args.out,
        args.calib,
        args.method,
        args.percentile,
        args.img_size
    )


if __name__ == "__main__":
    main()