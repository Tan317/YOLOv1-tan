"""
YOLOv1网络模型实现
==================

这个文件实现了YOLOv1目标检测网络的核心架构，包括：
1. 基于ResNet34的backbone网络
2. YOLOv1特有的检测头网络
3. 完整的损失函数计算
4. 网络前向传播逻辑

YOLOv1网络特点：
- 将输入图像划分为7x7网格
- 每个网格预测2个边界框和20个类别概率
- 输出维度：7x7x30 (30 = 5*2 + 20)
- 使用多任务损失函数进行端到端训练

作者：YOLOv1实现项目
"""

"""
YOLOv1网络模型实现
==================

这个文件实现了YOLOv1目标检测网络的核心架构，包括：
1. 基于ResNet34的backbone网络
2. YOLOv1特有的检测头网络
3. 完整的损失函数计算
4. 网络前向传播逻辑

YOLOv1网络特点：
- 将输入图像划分为7x7网格
- 每个网格预测2个边界框和20个类别概率
- 输出维度：7x7x30 (30 = 5*2 + 20)
- 使用多任务损失函数进行端到端训练

作者：YOLOv1实现项目
"""

import torch
import torch.nn as nn
import torchvision.models as tvmodel
from prepare_data import GL_CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import calculate_iou


class MyNet(nn.Module):
    """
    YOLOv1网络模型类
    ===============
    
    基于ResNet34的YOLOv1目标检测网络实现
    网络结构：
    1. ResNet34 backbone (去除最后两层)
    2. 4个卷积层 (1024通道)
    3. 2个全连接层
    4. 输出层 (7x7x30)
    """
    def __init__(self):
        """
        网络初始化函数
        ==============
        
        构建YOLOv1网络的完整架构：
        1. 加载预训练的ResNet34作为backbone
        2. 添加YOLOv1特有的检测头
        3. 设置输出层以匹配YOLOv1的输出格式
        """
        super(MyNet, self).__init__()
        
        # 第一步：加载预训练的ResNet34模型
        resnet = tvmodel.resnet34(pretrained=True)  # 使用预训练权重
        resnet_out_channel = resnet.fc.in_features  # 获取ResNet全连接层前的输出通道数
        
        # 去除ResNet的最后两层（平均池化和全连接层）
        # 只保留卷积层部分，用于特征提取
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        # 第二步：YOLOv1的检测头 - 4个卷积层
        # 这些卷积层将ResNet的特征图转换为YOLOv1所需的特征表示
        self.Conv_layers = nn.Sequential(
            # 第一个卷积层：调整通道数到1024
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 批归一化，加速训练收敛
            nn.LeakyReLU(inplace=True),  # 激活函数
            
            # 第二个卷积层：步长为2，降低特征图尺寸
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            
            # 第三个卷积层：保持特征图尺寸
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            
            # 第四个卷积层：保持特征图尺寸
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
        
        # 第三步：YOLOv1的全连接层
        # 将特征图转换为最终的检测结果
        self.Conn_layers = nn.Sequential(
            # 第一个全连接层：展平特征图并降维
            nn.Linear(GL_NUMGRID * GL_NUMGRID * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            
            # 第二个全连接层：输出最终的检测结果
            # 输出维度：7*7*(5*2+20) = 7*7*30
            nn.Linear(4096, GL_NUMGRID * GL_NUMGRID * (5*GL_NUMBBOX+len(GL_CLASSES))),
            nn.Sigmoid()  # Sigmoid激活，将输出限制在(0,1)范围内
        )

    def forward(self, inputs):
        """
        网络前向传播函数
        ===============
        
        参数:
            inputs: 输入图像张量，形状为(batch_size, 3, 448, 448)
        
        返回:
            pred: 网络预测结果，形状为(batch_size, 30, 7, 7)
        
        前向传播流程：
        1. ResNet backbone提取特征
        2. 卷积层进一步处理特征
        3. 全连接层生成最终预测
        4. 重塑输出为网格格式
        """
        # 第一步：通过ResNet backbone提取特征
        x = self.resnet(inputs)  # 输出形状：(batch_size, 512, 14, 14)
        
        # 第二步：通过YOLOv1的卷积层
        x = self.Conv_layers(x)  # 输出形状：(batch_size, 1024, 7, 7)
        
        # 第三步：展平特征图，准备输入全连接层
        x = x.view(x.size()[0], -1)  # 形状：(batch_size, 7*7*1024)
        
        # 第四步：通过全连接层生成最终预测
        x = self.Conn_layers(x)  # 形状：(batch_size, 7*7*30)
        
        # 第五步：重塑输出为网格格式，便于后续处理
        # 形状：(batch_size, 30, 7, 7)
        self.pred = x.reshape(-1, (5 * GL_NUMBBOX + len(GL_CLASSES)), GL_NUMGRID, GL_NUMGRID)
        
        return self.pred

    def calculate_loss(self, labels):
        """
        YOLOv1损失函数计算
        ==================
        
        YOLOv1使用多任务损失函数，包括：
        1. 坐标损失：边界框位置和大小的损失
        2. 置信度损失：包含物体和不包含物体的置信度损失
        3. 分类损失：物体类别的损失
        
        参数:
            labels: 真实标签，形状为(batch_size, 30, 7, 7)
        
        返回:
            loss: 总损失值
        
        损失函数组成：
        - 坐标损失：使用MSE损失，对宽高使用平方根
        - 置信度损失：使用MSE损失，包含物体的置信度目标为IoU
        - 分类损失：使用MSE损失
        """
        # 确保数据类型为double，提高计算精度
        self.pred = self.pred.double()
        labels = labels.double()
        
        # 网格数量
        num_gridx, num_gridy = GL_NUMGRID, GL_NUMGRID
        
        # 初始化各种损失
        noobj_confi_loss = 0.  # 不包含物体的网格的置信度损失
        coor_loss = 0.        # 包含物体的边界框坐标损失
        obj_confi_loss = 0.   # 包含物体的边界框置信度损失
        class_loss = 0.       # 包含物体的网格分类损失
        
        n_batch = labels.size()[0]  # 批次大小

        # 遍历每个批次样本
        for i in range(n_batch):
            # 遍历每个网格
            for n in range(num_gridx):  # x方向网格
                for m in range(num_gridy):  # y方向网格
                    
                    if labels[i, 4, m, n] == 1:  # 如果当前网格包含物体
                        # 将预测的边界框坐标转换为(x1,y1,x2,y2)格式，用于计算IoU
                        
                        # 第一个预测边界框的坐标转换
                        bbox1_pred_xyxy = (
                            (self.pred[i, 0, m, n] + n) / num_gridx - self.pred[i, 2, m, n] / 2,  # x1
                            (self.pred[i, 1, m, n] + m) / num_gridy - self.pred[i, 3, m, n] / 2,  # y1
                            (self.pred[i, 0, m, n] + n) / num_gridx + self.pred[i, 2, m, n] / 2,  # x2
                            (self.pred[i, 1, m, n] + m) / num_gridy + self.pred[i, 3, m, n] / 2   # y2
                        )
                        
                        # 第二个预测边界框的坐标转换
                        bbox2_pred_xyxy = (
                            (self.pred[i, 5, m, n] + n) / num_gridx - self.pred[i, 7, m, n] / 2,  # x1
                            (self.pred[i, 6, m, n] + m) / num_gridy - self.pred[i, 8, m, n] / 2,  # y1
                            (self.pred[i, 5, m, n] + n) / num_gridx + self.pred[i, 7, m, n] / 2,  # x2
                            (self.pred[i, 6, m, n] + m) / num_gridy + self.pred[i, 8, m, n] / 2   # y2
                        )
                        
                        # 真实边界框的坐标转换
                        bbox_gt_xyxy = (
                            (labels[i, 0, m, n] + n) / num_gridx - labels[i, 2, m, n] / 2,  # x1
                            (labels[i, 1, m, n] + m) / num_gridy - labels[i, 3, m, n] / 2,  # y1
                            (labels[i, 0, m, n] + n) / num_gridx + labels[i, 2, m, n] / 2,  # x2
                            (labels[i, 1, m, n] + m) / num_gridy + labels[i, 3, m, n] / 2   # y2
                        )
                        
                        # 计算两个预测边界框与真实边界框的IoU
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        
                        # 选择IoU较大的边界框作为负责预测该物体的边界框
                        if iou1 >= iou2:
                            # 第一个边界框负责预测
                            # 坐标损失：位置坐标使用MSE，宽高使用平方根MSE
                            coor_loss = coor_loss + 5 * (
                                torch.sum((self.pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) +  # 位置损失
                                torch.sum((self.pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2)  # 宽高损失
                            )
                            # 置信度损失：目标为IoU值
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 4, m, n] - iou1) ** 2
                            # 另一个边界框的置信度损失（目标为较小的IoU）
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            # 第二个边界框负责预测
                            coor_loss = coor_loss + 5 * (
                                torch.sum((self.pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) +  # 位置损失
                                torch.sum((self.pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2)  # 宽高损失
                            )
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 9, m, n] - iou2) ** 2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 4, m, n] - iou1) ** 2)
                        
                        # 分类损失：使用MSE损失
                        class_loss = class_loss + torch.sum((self.pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                        
                    else:  # 如果当前网格不包含物体
                        # 两个边界框的置信度都应该接近0
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(self.pred[i, [4, 9], m, n] ** 2)

        # 计算总损失
        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        
        # 返回平均损失
        return loss / n_batch

    def calculate_metric(self, preds, labels):
        """
        评估指标计算函数
        ================
        
        计算网络在验证集上的性能指标
        
        参数:
            preds: 网络预测结果
            labels: 真实标签
        
        返回:
            metric: 评估指标值
        """
        preds = preds.double()
        labels = labels[:, :(self.n_points*2)]
        l2_distance = torch.mean(torch.sum((preds-labels)**2, dim=1))
        return l2_distance


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    """
    网络测试代码
    ============
    
    用于验证网络结构是否正确，以及损失函数是否能正常计算
    """
    # 创建测试输入：5张448x448的RGB图像
    x = torch.zeros(5, 3, 448, 448)
    
    # 初始化网络
    net = MyNet()
    
    # 前向传播测试
    a = net(x)
    
    # 创建测试标签：5个样本，每个样本30个通道，7x7网格
    labels = torch.zeros(5, 30, 7, 7)
    
    # 计算损失
    loss = net.calculate_loss(labels)
    
    # 打印结果
    print("网络输出形状:", a.shape)
    print("损失值:", loss)
