from ultralytics import YOLOv10

# 加载预训练模型 (推荐用于迁移学习)
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
# 或者下载到本地后加载
# model = YOLOv10('yolov10s.pt')

# 开始训练
results = model.train(
    data='student_classroom.yaml',  # 您的数据集配置
    epochs=100,                     # 训练轮数
    batch=16,                       # 批次大小
    imgsz=640,                      # 图像尺寸
    patience=20,                    # 早停(防止过拟合)
    project='runs/classroom',       # 保存项目文件夹
    name='exp1',                    # 实验名称
    save=True,                      # 保存模型
    device=0                        # GPU设备
)

# 训练完成后评估模型
metrics = model.val()
print(f"模型性能: {metrics}")