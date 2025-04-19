import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLOv10
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from multiprocessing import freeze_support
import random
from ultralytics.nn.tasks import YOLOv10DetectionModel  # 注意不是 DetectionModel
import torch.serialization

torch.serialization.add_safe_globals([YOLOv10DetectionModel])

torch.serialization.add_safe_globals([DetectionModel])
torch.set_float32_matmul_precision('high')

# 设置随机种子以确保可重复性
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    freeze_support()
    seed_everything(42)

    # 增强数据增强策略，特别针对小目标检测
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0), p=0.7),  # 扩大缩放范围
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=20, p=0.7),  # 增加变换强度
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),  # 增加亮度对比度变化
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomFog(p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.OneOf([  # 添加更多随机变换
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),  # 添加Cutout增强
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 创建结果保存目录
    save_dir = 'runs/classroom/scb_dataset_improved'
    os.makedirs(save_dir, exist_ok=True)

    # 尝试加载模型
    try:
        model = YOLOv10.from_pretrained('jameslahm/yolov10s')
    except Exception as e:
        print(f"从 Hugging Face 加载失败: {e}")
        print("尝试加载本地模型 yolov10s.pt...")
        model = YOLOv10('E:/PycharmProjects/condaProjects/yolov10SCD/yolov10/yolov10s.pt')

    # 修改配置文件中的路径为相对路径
    yaml_file = 'student_classroom.yaml'
    with open(f'yolov10/{yaml_file}', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用相对路径替换绝对路径
    content = content.replace('E:/PycharmProjects/condaProjects/yolov10SCD/dataset', './dataset')
    
    # 保存修改后的配置文件
    with open(f'yolov10/student_classroom_rel.yaml', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已创建使用相对路径的配置文件")
    
    # 优化训练参数
    results = model.train(
        data='student_classroom_rel.yaml',
        epochs=300,  # 增加训练轮数
        batch=8,
        imgsz=640,
        patience=50,  # 增加早停耐心值以允许更长的训练
        project='runs/classroom',
        name='scb_dataset_improved',
        save=True,
        save_period=5,
        device=0,
        workers=4,  # 增加数据加载线程
        cos_lr=True,
        cache=True,
        amp=True,  # 使用混合精度训练
        close_mosaic=30,  # 进一步延迟关闭mosaic
        overlap_mask=True,
        lr0=0.002,  # 增大初始学习率
        lrf=0.001,  # 降低最终学习率因子，使学习率下降更慢
        warmup_epochs=15,  # 增加预热轮数
        weight_decay=0.0001,  # 调整权重衰减
        optimizer='AdamW',
        rect=True,
        box=8.0,  # 增加边界框损失权重
        cls=1.2,  # 增加类别损失权重
        dfl=1.8,  # 增加分布焦点损失权重
        # 新增参数
        mosaic=1.0,  # 使用mosaic增强
        mixup=0.3,   # 增加mixup增强
        copy_paste=0.3,  # 增加copy-paste增强
        auto_augment='randaugment',  # 使用随机增强
        erasing=0.5,  # 增加随机擦除概率
        crop_fraction=1.0,  # 使用随机裁剪
        label_smoothing=0.05,  # 添加标签平滑降低过拟合
        verbose=True,  # 详细输出训练过程
        multi_scale=True,  # 使用多尺度训练
        single_cls=False,  # 多类别检测
        deterministic=True,  # 确保结果可重复性
        seed=42,  # 设置随机种子
    )

    # 模型评估
    metrics = model.val()
    print(f"\n模型评估结果:\n{metrics}")

    # 各类别性能分析
    class_names = ["sit-forward", "read", "write", "hand-raising", "sleep", "talk"]
    print("\n各类别性能分析:")
    for i in range(6):
        prec = metrics.get(f'metrics/precision({i})', None)
        rec = metrics.get(f'metrics/recall({i})', None)
        f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec) > 0 else None
        if prec is not None and rec is not None:
            print(f"类别 {i} ({class_names[i]}): 精确率={prec:.4f}, 召回率={rec:.4f}, F1分数={f1:.4f if f1 else 'N/A'}")
    
    # 针对类别进行分析并输出低召回率类别的详细信息
    min_recall_idx = np.argmin([metrics.get(f'metrics/recall({i})', 1.0) for i in range(6)])
    print(f"\n⚠️ 召回率最低的类别是 {min_recall_idx} ({class_names[min_recall_idx]})")
    print("可能原因：1) 该类别样本较少 2) 特征不明显 3) 与其他类别混淆")
    print("建议：1) 增加该类别的数据量 2) 在验证集中检查该类别的标注质量")

    # 导出模型 - 多种格式
    print("\n导出模型...")
    model_path = f'{save_dir}/weights/best.pt'
    
    # 导出ONNX格式
    model.export(format='onnx', dynamic=True)
    print("✅ ONNX 模型导出成功")

    # 导出TorchScript格式
    try:
        model.export(format='torchscript')
        print("✅ TorchScript 模型导出成功")
    except Exception as e:
        print(f"⚠️ TorchScript导出失败: {e}")
    
    print("\n训练和评估完成！模型保存在:", save_dir)
    print("最佳性能: mAP50-95 =", metrics.get('metrics/mAP50-95', None))
    
    # 输出改进建议
    print("\n🔍 性能改进建议:")
    print("1. 针对低召回率的类别，考虑增加数据或使用类别平衡采样")
    print("2. 检查验证集中的样本是否代表真实场景")
    print("3. 尝试更大的模型架构或预训练权重")
    print("4. 考虑使用Focal Loss来处理类别不平衡问题")
