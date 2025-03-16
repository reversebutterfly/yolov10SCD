import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLOv10

# 确保 GPU 计算优化
torch.set_float32_matmul_precision('high')

# 数据增强配置
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转
    A.RandomBrightnessContrast(p=0.2),  # 20% 概率调整亮度和对比度
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 30% 概率进行高斯模糊
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),  # 平移、缩放、旋转
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化
    ToTensorV2()  # 转换为 PyTorch Tensor
])

# 加载 YOLOv10 预训练模型
model = YOLOv10.from_pretrained('jameslahm/yolov10s')

# 训练配置
results = model.train(
    data='student_classroom.yaml',  # 数据集配置文件
    epochs=200, batch=16, imgsz=640, patience=30,
    project='runs/classroom', name='exp2', save=True, save_period=10,
    device=0, workers=4, cos_lr=True, cache=True, amp=True,
    close_mosaic=10, overlap_mask=True, lr0=0.002, lrf=0.01,
    warmup_epochs=3, weight_decay=0.0005, optimizer='AdamW'
)

# 评估模型
metrics = model.val()
print(f"模型评估结果: {metrics}")

# 保存最终模型
model.export(format='onnx')
