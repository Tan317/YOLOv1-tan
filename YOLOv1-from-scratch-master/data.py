"""
YOLOv1数据集加载模块
====================

这个文件实现了YOLOv1训练所需的数据集类，包括：
1. 自定义Dataset类，继承自PyTorch的Dataset
2. 图像和标签的加载与预处理
3. 训练集和验证集的划分
4. 数据增强功能

数据集特点：
- 支持从txt文件加载图像路径
- 支持从csv文件加载标签数据
- 自动划分训练集和验证集
- 支持图像预处理和数据增强

作者：YOLOv1实现项目
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms


class MyDataset(Dataset):
    """
    YOLOv1自定义数据集类
    ====================
    
    继承自PyTorch的Dataset类，用于加载YOLOv1训练所需的数据
    支持训练集和验证集的自动划分，以及数据预处理功能
    """
    
    def __init__(self, dataset_dir, seed=None, mode="train", train_val_ratio=0.9, trans=None):
        """
        数据集初始化函数
        
        参数:
            dataset_dir: 数据集根目录路径
            seed: 随机种子，用于确保训练集和验证集划分的一致性
            mode: 数据集模式，"train"或"val"
            train_val_ratio: 训练集占总数据的比例（默认0.9）
            trans: 数据预处理和增强函数
        """
        # 设置随机种子，确保结果可复现
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
        
        # 保存数据集配置
        self.dataset_dir = dataset_dir
        self.mode = mode
        
        # 验证集和训练集使用相同的文件，只是索引不同
        if mode=="val":
            mode = "train"
            
        # 构建文件路径
        img_list_txt = os.path.join(dataset_dir, mode+".txt")  # 储存图片位置的列表文件
        label_csv = os.path.join(dataset_dir, mode+".csv")     # 储存标签的数组文件
        
        # 初始化数据列表
        self.img_list = []
        
        # 读取标签数据（CSV格式，每行是一个样本的标签向量）
        self.label = np.loadtxt(label_csv)
        
        # 读取图像路径列表
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
        
        # 数据集划分逻辑
        # 在mode=train或val时，将数据进行切分
        # 注意在mode="val"时，传入的随机种子seed要和mode="train"相同
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio*self.num_all_data)
        
        # 根据模式选择使用的数据索引
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]  # 使用前90%的数据作为训练集
        elif self.mode == "val":
            self.use_ids = all_ids[num_train:]  # 使用后10%的数据作为验证集
        else:
            self.use_ids = all_ids  # 使用全部数据

        # 保存数据预处理函数
        self.trans = trans

    def __len__(self):
        """获取数据集数量"""
        return len(self.use_ids)

    def __getitem__(self, item):
        """
        获取单个数据样本
        ================
        
        根据索引item返回对应的图像和标签数据
        
        参数:
            item: 数据索引
            
        返回:
            img: 预处理后的图像张量，形状为(C, H, W)
            label: 对应的标签张量
            
        处理步骤:
        1. 根据索引获取对应的图像路径和标签
        2. 加载图像并进行预处理
        3. 转换为PyTorch张量格式
        """
        # 获取对应的数据索引
        id = self.use_ids[item]
        
        # 获取标签数据并转换为张量
        label = torch.tensor(self.label[id, :])
        
        # 获取图像路径并加载图像
        img_path = self.img_list[id]
        img = Image.open(img_path)
        
        # 设置图像预处理流程
        if self.trans is None:
            # 默认预处理：转换为张量格式
            trans = transforms.Compose([
                # transforms.Resize((112,112)),  # 可选的resize操作
                transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
            ])
        else:
            trans = self.trans
            
        # 应用图像预处理和数据增强
        img = trans(img)
        
        # 调试代码（已注释）
        # transforms.ToPILImage()(img).show()  # 显示图像用于调试
        # print(label)  # 打印标签用于调试
        
        return img, label

if __name__ == '__main__':
    # 调试用，依次取出数据看看是否正确
    dataset_dir = r"C:\Users\xuanq\Desktop\VOC2012\voc2012_forYolov1"
    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, 1)
    for i in enumerate(dataloader):
        input("press enter to continue")