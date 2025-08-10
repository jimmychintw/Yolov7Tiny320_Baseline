#!/usr/bin/env python3
"""
ONNX 模型評測腳本
根據 Baseline Spec v1.0 的要求評測 mAP 和延遲
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YOLOv7ONNXEvaluator:
    """
    YOLOv7 ONNX 模型評測器
    """
    
    def __init__(self, model_path, img_size=320, device='cpu'):
        """
        Args:
            model_path: ONNX 模型路徑
            img_size: 輸入影像尺寸
            device: 推論裝置 ('cpu' 或 'cuda')
        """
        self.img_size = img_size
        self.device = device
        
        # 載入 ONNX 模型
        print(f"載入模型: {model_path}")
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"  輸入: {self.input_name}")
        print(f"  輸出: {self.output_names}")
    
    def preprocess(self, img_path):
        """
        影像預處理
        """
        # 讀取影像
        img = Image.open(img_path).convert('RGB')
        img_orig = np.array(img)
        
        # 記錄原始尺寸
        h0, w0 = img_orig.shape[:2]
        
        # Letterbox
        img = self.letterbox(img, new_shape=(self.img_size, self.img_size))
        
        # 轉換為 numpy array
        img = np.array(img, dtype=np.float32)
        
        # 正規化
        img = img / 255.0
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # 增加 batch 維度
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img, (h0, w0)
    
    def letterbox(self, img, new_shape=(320, 320), color=(114, 114, 114)):
        """
        YOLOv7 letterbox 實作
        """
        shape = img.size[::-1]  # PIL size is (width, height)
        
        # 計算縮放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        self.ratio = r  # 保存比例供後處理使用
        
        # 計算新尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # 分配 padding
        dw, dh = dw // 2, dh // 2
        self.pad = (dw, dh)  # 保存 padding 供後處理使用
        
        # 調整影像大小
        if shape[::-1] != new_unpad:
            img = img.resize(new_unpad, Image.BILINEAR)
        
        # 添加 padding
        top, bottom = dh, dh + (new_shape[0] - new_unpad[1]) % 2
        left, right = dw, dw + (new_shape[1] - new_unpad[0]) % 2
        
        img_padded = Image.new('RGB', new_shape[::-1], color)
        img_padded.paste(img, (left, top))
        
        return img_padded
    
    def postprocess(self, outputs, orig_shape, conf_thres=0.001, iou_thres=0.65, max_det=300):
        """
        後處理：NMS 和座標轉換
        
        Args:
            outputs: 模型輸出
            orig_shape: 原始影像尺寸 (h, w)
            conf_thres: 信心度閾值
            iou_thres: IoU 閾值
            max_det: 最大檢測數
        
        Returns:
            檢測結果 [[x1, y1, x2, y2, conf, cls], ...]
        """
        # 假設輸出格式為 [batch, num_boxes, 85] (x, y, w, h, obj_conf, cls_probs)
        predictions = outputs[0][0]  # 取第一個 batch
        
        # 過濾低信心度
        obj_conf = predictions[:, 4]
        valid_indices = obj_conf > conf_thres
        predictions = predictions[valid_indices]
        
        if len(predictions) == 0:
            return np.array([])
        
        # 計算類別信心度
        class_probs = predictions[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        class_confs = np.max(class_probs, axis=1)
        
        # 最終信心度 = obj_conf * class_conf
        scores = predictions[:, 4] * class_confs
        
        # 轉換 box 格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
        boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
        boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
        boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
        
        # 執行 NMS
        keep = self.nms(boxes, scores, iou_thres)
        
        # 限制最大檢測數
        if len(keep) > max_det:
            keep = keep[:max_det]
        
        # 組合結果
        detections = []
        for i in keep:
            # 座標縮放回原始尺寸
            box = boxes[i].copy()
            box[[0, 2]] = (box[[0, 2]] - self.pad[0]) / self.ratio
            box[[1, 3]] = (box[[1, 3]] - self.pad[1]) / self.ratio
            
            # 裁剪到影像邊界
            box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_shape[1])
            box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_shape[0])
            
            detections.append([
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(scores[i]), int(class_ids[i])
            ])
        
        return np.array(detections)
    
    def nms(self, boxes, scores, iou_thres):
        """
        Non-Maximum Suppression
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        
        return keep
    
    def evaluate_coco(self, val_list_path, ann_file, conf_thres=0.001, iou_thres=0.65):
        """
        在 COCO 驗證集上評測 mAP
        """
        print("\n=== COCO mAP 評測 ===")
        
        # 載入 COCO 標註
        coco_gt = COCO(ann_file)
        
        # 讀取驗證集清單
        with open(val_list_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        # 推論並收集結果
        results = []
        print(f"處理 {len(image_paths)} 張影像...")
        
        for img_path in tqdm(image_paths):
            # 獲取影像 ID
            img_name = Path(img_path).stem
            img_id = int(img_name)
            
            # 預處理
            img_tensor, orig_shape = self.preprocess(img_path)
            
            # 推論
            outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
            
            # 後處理
            detections = self.postprocess(outputs, orig_shape, conf_thres, iou_thres)
            
            # 轉換為 COCO 格式
            for det in detections:
                x1, y1, x2, y2, score, cls_id = det
                w = x2 - x1
                h = y2 - y1
                
                results.append({
                    'image_id': img_id,
                    'category_id': int(cls_id) + 1,  # COCO 類別 ID 從 1 開始
                    'bbox': [x1, y1, w, h],
                    'score': score
                })
        
        # 計算 mAP
        if len(results) > 0:
            coco_dt = coco_gt.loadRes(results)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 提取指標
            metrics = {
                'mAP50-95': coco_eval.stats[0],
                'mAP50': coco_eval.stats[1],
                'mAP75': coco_eval.stats[2],
                'mAP50-95_small': coco_eval.stats[3],
                'mAP50-95_medium': coco_eval.stats[4],
                'mAP50-95_large': coco_eval.stats[5]
            }
        else:
            metrics = {
                'mAP50-95': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP50-95_small': 0.0,
                'mAP50-95_medium': 0.0,
                'mAP50-95_large': 0.0
            }
        
        return metrics
    
    def benchmark_latency(self, warmup=50, iterations=200):
        """
        測量推論延遲
        """
        print(f"\n=== 延遲測試 ===")
        print(f"預熱: {warmup} 次")
        print(f"測試: {iterations} 次")
        
        # 建立隨機輸入
        dummy_input = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        
        # 預熱
        print("預熱中...")
        for _ in range(warmup):
            _ = self.session.run(self.output_names, {self.input_name: dummy_input})
        
        # 測量延遲
        print("測量中...")
        latencies = []
        for _ in tqdm(range(iterations)):
            start_time = time.perf_counter()
            _ = self.session.run(self.output_names, {self.input_name: dummy_input})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # 轉換為毫秒
        
        latencies = np.array(latencies)
        
        # 計算統計值
        stats = {
            'mean': float(np.mean(latencies)),
            'median': float(np.median(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99))
        }
        
        print(f"\n延遲統計 (ms):")
        print(f"  平均值: {stats['mean']:.2f}")
        print(f"  中位數: {stats['median']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='ONNX 模型評測')
    parser.add_argument('--model', type=str, required=True,
                        help='ONNX 模型路徑')
    parser.add_argument('--val-list', type=str, default='val.txt',
                        help='驗證集清單檔案')
    parser.add_argument('--ann-file', type=str, 
                        default='/data/coco/annotations/instances_val2017.json',
                        help='COCO 標註檔案')
    parser.add_argument('--img', '--img-size', type=int, default=320,
                        help='輸入影像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='信心度閾值')
    parser.add_argument('--iou-thres', type=float, default=0.65,
                        help='NMS IoU 閾值')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='推論裝置')
    parser.add_argument('--report', type=str, default='eval_report.json',
                        help='輸出報告檔案')
    parser.add_argument('--append', action='store_true',
                        help='附加到現有報告')
    parser.add_argument('--skip-map', action='store_true',
                        help='跳過 mAP 評測')
    parser.add_argument('--skip-latency', action='store_true',
                        help='跳過延遲測試')
    
    args = parser.parse_args()
    
    # 檢查檔案
    if not os.path.exists(args.model):
        print(f"錯誤: 找不到模型檔案 {args.model}")
        sys.exit(1)
    
    # 建立評測器
    evaluator = YOLOv7ONNXEvaluator(args.model, args.img, args.device)
    
    # 準備報告
    report = {}
    if args.append and os.path.exists(args.report):
        with open(args.report, 'r') as f:
            report = json.load(f)
    
    model_name = Path(args.model).name
    report[model_name] = {
        'model': model_name,
        'img_size': args.img,
        'device': args.device
    }
    
    # mAP 評測
    if not args.skip_map:
        metrics = evaluator.evaluate_coco(
            args.val_list,
            args.ann_file,
            args.conf_thres,
            args.iou_thres
        )
        report[model_name]['metrics'] = metrics
    
    # 延遲測試
    if not args.skip_latency:
        latency = evaluator.benchmark_latency()
        report[model_name]['latency'] = latency
    
    # 儲存報告
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n報告已儲存至: {args.report}")


if __name__ == "__main__":
    main()