"""
YOLOv1测试脚本
==============

这个文件实现了YOLOv1网络的测试和推理功能，包括：
1. 模型加载和推理
2. 检测结果可视化
3. 边界框绘制
4. 置信度显示

测试流程：
1. 加载训练好的模型
2. 读取测试图像
3. 执行目标检测
4. 绘制检测结果
5. 显示检测图像

作者：YOLOv1实现项目
"""

import os
from my_arguments import Args
import torch
from torch.utils.data import DataLoader

from model import MyNet
from data import MyDataset
from util import labels2bbox
from prepare_data import GL_CLASSES
import torchvision.transforms as transforms
from PIL import Image
import cv2


COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),]  # 用来标识20个类别的bbox颜色，可自行设定


class TestInterface(object):
    """
    YOLOv1测试接口类
    ===============
    
    负责模型的测试和推理功能，包括：
    - 模型加载和推理
    - 检测结果可视化
    - 边界框绘制
    - 置信度显示
    """
    
    def __init__(self, opts):
        """
        测试接口初始化
        
        参数:
            opts: 测试参数配置对象
        """
        self.opts = opts
        print("=======================Start inferring.=======================")

    def main(self):
        """
        YOLOv1测试主函数
        ===============
        
        执行完整的测试流程：
        1. 加载训练好的模型
        2. 读取测试图像
        3. 执行目标检测
        4. 绘制检测结果
        5. 显示检测图像
        
        测试流程：
        1. 获取命令行参数
        2. 获取测试图像列表
        3. 加载网络模型
        4. 对每张图像进行推理
        5. 可视化检测结果
        """
        opts = self.opts
        
        # 获取测试图像列表
        img_list = os.listdir(opts.dataset_dir)
        
        # 设置图像预处理流程
        trans = transforms.Compose([
            # transforms.Resize((112, 112)),  # 可选的resize操作
            transforms.ToTensor(),  # 转换为张量并归一化
        ])
        
        # 加载训练好的模型
        model = torch.load(opts.weight_path)
        if opts.use_GPU:
            model.to(opts.GPU_id)
            
        # 对每张测试图像进行推理
        for img_name in img_list:
            # 读取图像
            img_path = os.path.join(opts.dataset_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            
            # 图像预处理
            img = trans(img)
            img = torch.unsqueeze(img, dim=0)  # 添加batch维度
            print(img_name, img.shape)
            
            # 移动到GPU（如果使用GPU）
            if opts.use_GPU:
                img = img.to(opts.GPU_id)
                
            # 模型推理
            preds = torch.squeeze(model(img), dim=0).detach().cpu()  # 移除batch维度并转换为CPU
            preds = preds.permute(1,2,0)  # 调整维度顺序为(H,W,C)
            
            # 将网络输出转换为边界框格式
            bbox = labels2bbox(preds)
            
            # 读取原始图像用于绘制
            draw_img = cv2.imread(img_path)
            
            # 绘制检测结果
            self.draw_bbox(draw_img, bbox)

    def draw_bbox(self, img, bbox):
        """
        在图像上绘制检测到的边界框
        ============================
        
        根据检测结果在图像上绘制边界框、类别名称和置信度
        
        参数:
            img: 要绘制边界框的图像
            bbox: 检测结果数组，形状为(n,6)
                  每行包含：(x1,y1,x2,y2, confidence, class_id)
                  其中坐标是归一化的(0-1范围)
        """
        # 获取图像尺寸
        h, w = img.shape[0:2]
        n = bbox.shape[0]  # 检测到的边界框数量
        
        # 遍历每个检测到的边界框
        for i in range(n):
            # 获取置信度
            confidence = bbox[i, 4]
            
            # 过滤低置信度的检测结果
            if confidence < 0.2:
                continue
                
            # 将归一化坐标转换为像素坐标
            p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))  # 左上角点
            p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))  # 右下角点
            
            # 获取类别名称
            cls_name = GL_CLASSES[int(bbox[i, 5])]
            print(cls_name, p1, p2)
            
            # 绘制边界框矩形
            cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])
            
            # 绘制类别名称
            cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            # 绘制置信度分数
            cv2.putText(img, str(confidence), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
        # 显示检测结果图像
        cv2.imshow("bbox", img)
        cv2.waitKey(0)  # 等待按键继续


if __name__ == '__main__':
    # 网络测试代码
    args = Args()
    args.set_test_args()  # 获取命令行参数
    test_interface = TestInterface(args.get_opts())
    test_interface.main()  # 调用测试接口