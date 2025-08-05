"""
YOLOv1参数配置模块
==================

这个文件定义了YOLOv1项目所需的所有命令行参数，包括：
1. 训练参数配置
2. 测试参数配置
3. 模型和数据集路径
4. 硬件设备配置

主要功能：
- 统一管理项目参数
- 支持命令行参数解析
- 自动检测GPU可用性
- 提供默认参数配置

作者：YOLOv1实现项目
"""

import argparse
import torch

class Args(object):
    """
    YOLOv1参数配置类
    ===============
    
    负责管理YOLOv1项目的所有命令行参数，包括训练和测试阶段的参数配置
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        """
        设置训练参数
        ============
        
        配置YOLOv1训练阶段所需的所有命令行参数，包括：
        - 训练超参数（学习率、批次大小等）
        - 数据集和模型路径
        - 硬件设备配置
        - 训练控制参数
        """
        # 训练超参数
        self.parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
        self.parser.add_argument("--lr", type=float, default=0.001, help="学习率")
        self.parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减系数")
        
        # 训练轮数设置
        self.parser.add_argument("--epoch", type=int, default=60, help="总训练轮数")
        self.parser.add_argument("--start_epoch", type=int, default=19, help="开始训练的轮数（用于断点续训）")
        
        # GPU配置
        self.parser.add_argument("--use_GPU", action="store_true", help="是否使用GPU训练")
        self.parser.add_argument("--GPU_id", type=int, default=None, help="GPU设备ID")
        
        # 数据集和模型路径
        self.parser.add_argument("--dataset_dir", type=str, default=r"C:\Users\xuanq\Desktop\VOC2012\voc2012_forYolov1", help="数据集目录路径")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="模型保存目录")
        
        # 训练控制参数
        self.parser.add_argument("--print_freq", type=int, default=20, help="打印训练信息的频率（每n次迭代）")
        self.parser.add_argument("--save_freq", type=int, default=1, help="保存模型的频率（每n轮）")
        self.parser.add_argument("--num_workers", type=int, default=4, help="数据加载的线程数")
        
        # 预训练模型和随机种子
        self.parser.add_argument("--pretrain", type=str, default=r"D:\My_project\deeplearning\yolov1\checkpoints\epoch18.pkl", help="预训练模型路径")
        self.parser.add_argument("--random_seed", type=int, default=0, help="随机种子（用于数据集划分）")

        # 解析命令行参数
        self.opts = self.parser.parse_args()

        # 自动检测GPU可用性
        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def set_test_args(self):
        """
        设置测试参数
        ============
        
        配置YOLOv1测试阶段所需的所有命令行参数，包括：
        - 测试超参数
        - 模型权重路径
        - 测试图像路径
        - 硬件设备配置
        """
        # 测试超参数
        self.parser.add_argument("--batch_size", type=int, default=1, help="测试批次大小")
        
        # GPU配置
        self.parser.add_argument("--use_GPU", action="store_true", help="是否使用GPU进行推理")
        self.parser.add_argument("--GPU_id", type=int, default=None, help="GPU设备ID")
        
        # 测试数据路径
        self.parser.add_argument("--dataset_dir", type=str, default=r"C:\Users\xuanq\Desktop\VOC2012\voc2012_forYolov1\img", help="测试图像目录路径")
        
        # 模型权重路径
        self.parser.add_argument("--weight_path", type=str,
                            default=r"D:\My_project\deeplearning\yolov1\checkpoints\epoch18.pkl",
                            help="模型权重文件路径")

        # 解析命令行参数
        self.opts = self.parser.parse_args()
        
        # 自动检测GPU可用性
        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def get_opts(self):
        """
        获取解析后的参数对象
        
        返回:
            opts: 包含所有解析后参数的命名空间对象
        """
        return self.opts