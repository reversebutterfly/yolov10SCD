# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import os
import torch
from ultralytics import YOLOv10
import argparse
from pathlib import Path

def train_model(args):
    """训练模型"""
    print("开始训练模型...")
    
    # 创建模型
    model = YOLOv10(args.model_path)
    
    # 训练参数
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        patience=args.patience,
        project=args.project,
        name=args.name,
        save=True,
        save_period=5,
        device=args.device,
        workers=args.workers,
        cos_lr=True,
        cache=True,
        amp=True,
        close_mosaic=15,
        overlap_mask=True,
        lr0=args.learning_rate,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,
        optimizer='AdamW',
        rect=True,
        box=7.5,
        cls=0.8,
        dfl=1.5
    )
    
    return model

def evaluate_model(model, args):
    """评估模型"""
    print("\n开始评估模型...")
    metrics = model.val()
    print(f"\n模型评估结果:\n{metrics}")
    
    # 各类别性能分析
    class_names = ["sit-forward", "read", "write", "hand-raising", "sleep", "talk"]
    print("\n各类别性能分析:")
    for i in range(6):
        prec = metrics.get(f'metrics/precision({i})', None)
        rec = metrics.get(f'metrics/recall({i})', None)
        if prec is not None and rec is not None:
            print(f"类别 {i} ({class_names[i]}): 精确率 = {prec:.4f}, 召回率 = {rec:.4f}")
    
    return metrics

def export_model(model, args):
    """导出模型"""
    print("\n开始导出模型...")
    # 导出ONNX
    model.export(format='onnx', dynamic=True)
    print("✅ ONNX 模型导出成功")
    
    # 导出TorchScript
    try:
        model.export(format='torchscript')
        print("✅ TorchScript 模型导出成功")
    except Exception as e:
        print(f"⚠️ TorchScript导出失败: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv10 课堂行为检测训练和评估')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'export'],
                        help='运行模式: train, eval, export')
    parser.add_argument('--model_path', type=str, default='yolov10/yolov10s.pt',
                        help='模型路径')
    parser.add_argument('--data_yaml', type=str, default='yolov10/student_classroom.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=120,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--img_size', type=int, default=640,
                        help='图像大小')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    parser.add_argument('--project', type=str, default='runs/classroom',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default='scb_dataset5',
                        help='实验名称')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备')
    parser.add_argument('--workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='初始学习率')
    return parser.parse_args()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.project, exist_ok=True)
    
    if args.mode == 'train':
        # 训练模型
        model = train_model(args)
        
        # 评估模型
        evaluate_model(model, args)
        
        # 导出模型
        export_model(model, args)
    
    elif args.mode == 'eval':
        # 加载模型
        model = YOLOv10(args.model_path)
        
        # 评估模型
        evaluate_model(model, args)
    
    elif args.mode == 'export':
        # 加载模型
        model = YOLOv10(args.model_path)
        
        # 导出模型
        export_model(model, args)

if __name__ == '__main__':
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
