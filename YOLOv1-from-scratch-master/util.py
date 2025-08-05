"""
YOLOv1工具函数模块
==================

这个文件提供了YOLOv1项目所需的各种工具函数和类，包括：
1. IoU计算函数
2. 边界框格式转换
3. 非极大值抑制(NMS)算法
4. 网络层工具类
5. 后处理函数

主要功能：
- 计算两个边界框的IoU
- 将网络输出转换为边界框格式
- 执行NMS算法去除重复检测
- 提供深度可分离卷积等网络层

作者：YOLOv1实现项目
"""

import torch
import torch.nn as nn
import numpy as np


class DepthwiseConv(nn.Module):
    """
    深度可分离卷积层5
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(DepthwiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class InvertedBottleneck(nn.Module):
    """
    MobileNet v2 的InvertedBottleneck
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, t_factor, padding=0, bias=True):
        super(InvertedBottleneck, self).__init__()
        mid_channels = t_factor*in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs

class Flatten(nn.Module):
    """
    将三维张量拉平的网络层
    (n,c,h,w) -> (n, c*h*w)
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)
        return x


def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的IoU (Intersection over Union)
    ==============================================
    
    IoU = 交集面积 / 并集面积
    
    参数:
        bbox1: 第一个边界框，格式为(x1, y1, x2, y2)
        bbox2: 第二个边界框，格式为(x1, y1, x2, y2)
    
    返回:
        iou: IoU值，范围[0, 1]
    
    计算步骤:
    1. 检查边界框的有效性
    2. 计算交集区域的坐标
    3. 计算交集面积
    4. 计算IoU值
    """
    # 检查边界框的有效性：确保x2>x1且y2>y1
    if bbox1[2]<=bbox1[0] or bbox1[3]<=bbox1[1] or bbox2[2]<=bbox2[0] or bbox2[3]<=bbox2[1]:
        return 0  # 如果bbox1或bbox2没有面积，或者输入错误，直接返回0

    # 计算交集区域的坐标
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的重合区域的(x1,y1,x2,y2)
    
    # 交集区域的左上角坐标：取两个边界框左上角坐标的最大值
    intersect_bbox[0] = max(bbox1[0], bbox2[0])  # x1
    intersect_bbox[1] = max(bbox1[1], bbox2[1])  # y1
    
    # 交集区域的右下角坐标：取两个边界框右下角坐标的最小值
    intersect_bbox[2] = min(bbox1[2], bbox2[2])  # x2
    intersect_bbox[3] = min(bbox1[3], bbox2[3])  # y2

    # 计算交集区域的宽度和高度
    w = max(intersect_bbox[2] - intersect_bbox[0], 0)  # 宽度，如果无交集则为0
    h = max(intersect_bbox[3] - intersect_bbox[1], 0)  # 高度，如果无交集则为0
    
    # 计算两个边界框的面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    
    # 计算交集面积
    area_intersect = w * h
    
    # 计算IoU：交集面积 / 并集面积
    # 并集面积 = 面积1 + 面积2 - 交集面积
    # 加上1e-6防止除零错误
    iou = area_intersect / (area1 + area2 - area_intersect + 1e-6)
    
    # 调试代码（已注释）
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()
    
    return iou


def labels2bbox(matrix):
    """
    将YOLOv1网络输出转换为边界框格式并执行NMS
    ============================================
    
    将网络输出的7*7*30数据转换为边界框格式，然后执行NMS算法去除重复检测
    
    参数:
        matrix: 网络输出张量，形状为(7,7,30)
               注意：输入数据中bbox坐标格式为(px,py,w,h)，需要转换为(x1,y1,x2,y2)格式
    
    返回:
        bboxes: NMS处理后的边界框，形状为(-1, 6)
                每行包含：(x1,y1,x2,y2, confidence, class_id)
    
    处理步骤:
    1. 检查输入数据格式
    2. 将7*7*30数据转换为98个边界框
    3. 坐标格式转换：从相对坐标转换为绝对坐标
    4. 执行NMS算法去除重复检测
    """
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size: ", matrix.size(), " != (7,7)")
    matrix = matrix.numpy()
    bboxes = np.zeros((98, 6))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    matrix = matrix.reshape(49,-1)
    bbox = matrix[:, :10].reshape(98, 5)
    r_grid = np.array(list(range(7)))
    r_grid = np.repeat(r_grid, repeats=14, axis=0)  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]
    c_grid = np.array(list(range(7)))
    c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]
    c_grid = np.repeat(c_grid, repeats=7, axis=0).reshape(-1)  # [0 0 1 1 2 2 3 3 4 4 5 5 6 6 0 0 1 1 2 2 3 3 4 4 5 5 6 6...]
    bboxes[:, 0] = np.maximum((bbox[:, 0] + c_grid) / 7.0 - bbox[:, 2] / 2.0, 0)
    bboxes[:, 1] = np.maximum((bbox[:, 1] + r_grid) / 7.0 - bbox[:, 3] / 2.0, 0)
    bboxes[:, 2] = np.minimum((bbox[:, 0] + c_grid) / 7.0 + bbox[:, 2] / 2.0, 1)
    bboxes[:, 3] = np.minimum((bbox[:, 1] + r_grid) / 7.0 + bbox[:, 3] / 2.0, 1)
    bboxes[:, 4] = bbox[:, 4]
    cls = np.argmax(matrix[:, 10:], axis=1)
    cls = np.repeat(cls, repeats=2, axis=0)
    bboxes[:, 5] = cls
    # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox
    keepid = nms_multi_cls(bboxes, thresh=0.1, n_cls=20)
    ids = []
    for x in keepid:
        ids = ids + list(x)
    ids = sorted(ids)
    return bboxes[ids, :]


def nms_1cls(dets, thresh):
    """
    单类别非极大值抑制算法 (Non-Maximum Suppression)
    ================================================
    
    去除同一类别中重叠度高的检测框，保留置信度最高的检测框
    
    参数:
        dets: 检测结果数组，形状为(N, 5)，每行包含(x1,y1,x2,y2,score)
        thresh: IoU阈值，超过此阈值的检测框会被抑制
    
    返回:
        keep: 保留的检测框索引列表
    
    算法步骤:
    1. 按置信度降序排序所有检测框
    2. 选择置信度最高的检测框
    3. 计算其他检测框与当前检测框的IoU
    4. 抑制IoU超过阈值的检测框
    5. 重复步骤2-4直到处理完所有检测框
    """
    # 从检测结果dets中获得x1,y1,x2,y2和scores的值
    x1 = dets[:, 0]  # 左上角x坐标
    y1 = dets[:, 1]  # 左上角y坐标
    x2 = dets[:, 2]  # 右下角x坐标
    y2 = dets[:, 3]  # 右下角y坐标
    scores = dets[:, 4]  # 置信度分数

    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 按照置信度score的值降序排序的下标序列
    order = scores.argsort()[::-1]

    # keep用来保存最后保留的检测框的下标
    keep = []
    
    while order.size > 0:
        # 当前置信度最高bbox的index
        i = order[0]
        
        # 添加当前剩余检测框中得分最高的index到keep中
        keep.append(i)
        
        # 得到此bbox和剩余其他bbox的相交区域，左上角和右下角
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交集区域左上角x坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交集区域左上角y坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交集区域右下角x坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交集区域右下角y坐标

        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)  # 交集区域宽度
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 交集区域高度
        inter = w * h  # 交集面积
        
        # 计算IoU：重叠面积/(面积1+面积2-重叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的bbox（抑制重叠度高的检测框）
        inds = np.where(iou <= thresh)[0]
        order = order[inds+1]  # 更新剩余检测框列表
        
    return keep


def nms_multi_cls(dets, thresh, n_cls):
    """
    多类别非极大值抑制算法
    =====================
    
    对多个类别的检测结果分别执行NMS算法
    
    参数:
        dets: 检测结果数组，形状为(N, 6)，每行包含(x1,y1,x2,y2,score,class_id)
        thresh: IoU阈值
        n_cls: 类别数量
    
    返回:
        keeps_index: 每个类别保留的检测框索引列表
                    keeps_index[i]表示第i类保留下来的bbox下标list
    """
    # 储存结果的列表，keeps_index[i]表示第i类保留下来的bbox下标list
    keeps_index = []
    
    # 对每个类别分别执行NMS
    for i in range(n_cls):
        # 找到属于当前类别的所有检测框
        order_i = np.where(dets[:,5]==i)[0]  # 当前类别的检测框索引
        
        # 提取当前类别的检测框数据（只包含坐标和置信度）
        det = dets[dets[:, 5] == i, 0:5]
        
        # 如果当前类别没有检测框，添加空列表
        if det.shape[0] == 0:
            keeps_index.append([])
            continue
            
        # 对当前类别执行单类别NMS
        keep = nms_1cls(det, thresh)
        
        # 将NMS结果添加到结果列表
        keeps_index.append(order_i[keep])
        
    return keeps_index


if __name__ == '__main__':
    a = torch.randn((7,7,30))
    print(a)
    labels2bbox(a)