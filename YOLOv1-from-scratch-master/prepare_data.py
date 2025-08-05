"""
数据准备脚本 - YOLOv1目标检测项目
====================================

这个脚本的主要功能是将VOC2012数据集处理成YOLOv1训练所需的格式。

处理后的数据包括：
1. train.csv/test.csv: 每行是一张图片的标签数据，格式为(7*7*30)的一维向量
2. train.txt/test.txt: 每行是图片的路径，与csv文件中的标签一一对应

数据预处理步骤：
1. 将XML标注文件转换为txt格式的边界框信息
2. 对图像进行padding和resize处理，统一尺寸为448x448
3. 将边界框坐标转换为YOLOv1所需的网格格式
4. 生成训练和测试数据集

作者：YOLOv1实现项目
"""

import xml.etree.ElementTree as ET  # 用于解析XML标注文件
import numpy as np  # 数值计算库
import cv2  # 图像处理库
import random  # 随机数生成
import os  # 操作系统接口


# ==================== 全局配置参数 ====================
# VOC数据集的20个类别名称
GL_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

GL_NUMBBOX = 2  # 每个网格预测的边界框数量（YOLOv1中为2）
GL_NUMGRID = 7  # 图像被划分为7x7的网格
STATIC_DATASET_PATH = r'C:\Users\xuanq\Desktop\VOC2012'  # 数据集路径
STATIC_DEBUG = False  # 调试模式开关


def convert(size, box):
    """
    坐标格式转换函数
    =================
    
    将边界框的坐标从(左上角x, 右下角x, 左上角y, 右下角y)格式
    转换为(中心点x, 中心点y, 宽度, 高度)格式，并进行归一化
    
    参数:
        size: 图像尺寸 (width, height)
        box: 边界框坐标 (xmin, xmax, ymin, ymax)
    
    返回:
        归一化后的边界框坐标 (x_center, y_center, width, height)
    """
    dw = 1. / size[0]  # 宽度的归一化系数
    dh = 1. / size[1]  # 高度的归一化系数
    
    # 计算边界框中心点坐标
    x = (box[0] + box[1]) / 2.0  # 中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 中心点y坐标
    
    # 计算边界框的宽度和高度
    w = box[1] - box[0]  # 宽度
    h = box[3] - box[2]  # 高度
    
    # 归一化坐标（除以图像尺寸）
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)


def convert_annotation(anno_dir, image_id, labels_dir):
    """
    XML标注文件转换函数
    ===================
    
    将单个图像的XML标注文件转换为YOLOv1训练所需的txt格式
    
    参数:
        anno_dir: 标注文件目录
        image_id: 图像文件名（包含.xml扩展名）
        labels_dir: 输出标签文件目录
    
    处理过程:
        1. 解析XML文件，提取图像尺寸和物体信息
        2. 过滤掉困难样本和不在类别列表中的物体
        3. 将边界框坐标转换为归一化格式
        4. 保存为txt文件，格式：类别ID x_center y_center width height
    """
    # 打开XML标注文件
    in_file = open(os.path.join(anno_dir, 'Annotations/%s' % (image_id)))
    image_id = image_id.split('.')[0]  # 去掉.xml扩展名
    
    # 解析XML文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # 遍历XML中的所有物体
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text  # 是否为困难样本
        cls = obj.find('name').text  # 物体类别名称
        
        # 过滤条件：类别不在列表中或为困难样本
        if cls not in GL_CLASSES or int(difficult) == 1:
            continue
            
        cls_id = GL_CLASSES.index(cls)  # 获取类别ID
        
        # 提取边界框坐标
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        
        # 转换坐标格式并归一化
        bb = convert((w, h), points)
        
        # 保存到txt文件
        with open(os.path.join(labels_dir, '%s.txt' % (image_id)), 'a') as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def make_label_txt(anno_dir, labels_dir):
    """
    批量转换标注文件
    ================
    
    将anno_dir目录下所有图像的XML标注文件转换为txt格式
    
    参数:
        anno_dir: 包含Annotations文件夹的目录
        labels_dir: 输出标签文件的目录
    """
    # 获取Annotations文件夹中的所有XML文件
    filenames = os.listdir(os.path.join(anno_dir,'Annotations'))
    
    # 逐个转换每个文件的标注
    for file in filenames:
        convert_annotation(anno_dir, file, labels_dir)


def img_augument(img_dir, save_img_dir, labels_dir):
    """
    图像预处理和增强函数
    ====================
    
    对图像进行预处理，包括padding和resize操作，同时调整对应的边界框坐标
    
    参数:
        img_dir: 原始图像目录
        save_img_dir: 处理后图像保存目录
        labels_dir: 标签文件目录
    
    处理步骤:
        1. 读取原始图像
        2. 进行padding操作，使图像变为正方形
        3. 将图像resize到448x448
        4. 根据padding调整边界框坐标
        5. 保存处理后的图像和标签
    """
    # 获取所有有标签的图像文件名
    imgs_list = [x.split('.')[0]+".jpg" for x in os.listdir(labels_dir)]
    
    for img_name in imgs_list:
        print("process %s"%os.path.join(img_dir, img_name))
        
        # 读取原始图像
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[0:2]  # 获取图像高度和宽度
        
        input_size = 448  # YOLOv1网络的输入尺寸
        
        # Padding操作：将非正方形图像填充为正方形
        padw, padh = 0, 0  # 记录宽高方向的padding数值
        
        if h > w:  # 高度大于宽度，需要在宽度方向padding
            padw = (h - w) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
        elif w > h:  # 宽度大于高度，需要在高度方向padding
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
            
        # 将图像resize到448x448
        img = cv2.resize(img, (input_size, input_size))
        
        # 保存处理后的图像
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)
        
        # 读取对应的边界框信息
        with open(os.path.join(labels_dir,img_name.split('.')[0] + ".txt"), 'r') as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        
        # 验证边界框数据格式是否正确
        if len(bbox) % 5 != 0:
            raise ValueError("File:"
                             + os.path.join(labels_dir,img_name.split('.')[0] + ".txt") + "——bbox Extraction Error!")

        # 根据padding调整边界框坐标
        if padw != 0:  # 如果进行了宽度方向的padding
            for i in range(len(bbox) // 5):
                # 调整x坐标和宽度
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
                
                # 调试模式：在图像上绘制边界框
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        elif padh != 0:  # 如果进行了高度方向的padding
            for i in range(len(bbox) // 5):
                # 调整y坐标和高度
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
                
                # 调试模式：在图像上绘制边界框
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        
        # 调试模式：显示处理后的图像
        if STATIC_DEBUG:
            cv2.imshow("bbox-%d"%int(bbox[0]), img)
            cv2.waitKey(0)
            
        # 保存调整后的边界框信息
        with open(os.path.join(labels_dir, img_name.split('.')[0] + ".txt"), 'w') as f:
            for i in range(len(bbox) // 5):
                bbox = [str(x) for x in bbox[i*5:(i*5+5)]]
                str_context = " ".join(bbox)+'\n'
                f.write(str_context)


def convert_bbox2labels(bbox):
    """
    边界框转换为YOLOv1标签格式
    ==========================
    
    将边界框的(cls,x,y,w,h)数据转换为YOLOv1训练所需的网格格式
    输入格式：(类别ID, 中心点x, 中心点y, 宽度, 高度)
    输出格式：(7,7,30)的网格标签，其中30 = 5*2 + 20（2个边界框 + 20个类别）
    
    参数:
        bbox: 边界框数据列表，每5个元素表示一个边界框
    
    返回:
        转换后的标签数据，形状为(1, 7*7*30)
    """
    gridsize = 1.0/GL_NUMGRID  # 每个网格的大小（1/7）
    
    # 初始化标签数组：(7,7,30) - 7x7网格，每个网格30个值
    labels = np.zeros((7,7,5*GL_NUMBBOX+len(GL_CLASSES)))
    
    # 处理每个边界框
    for i in range(len(bbox)//5):
        # 计算边界框中心点所在的网格位置
        gridx = int(bbox[i*5+1] // gridsize)  # 列索引
        gridy = int(bbox[i*5+2] // gridsize)  # 行索引
        
        # 计算边界框中心点相对于网格左上角的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx  # 相对x坐标
        gridpy = bbox[i * 5 + 2] / gridsize - gridy  # 相对y坐标
        
        # 设置第一个边界框的信息：(相对x, 相对y, 宽度, 高度, 置信度)
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        
        # 设置第二个边界框的信息（YOLOv1中每个网格预测2个边界框）
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        
        # 设置类别概率（one-hot编码）
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
        
    # 将标签展平为一维向量
    labels = labels.reshape(1, -1)
    return labels


def create_csv_txt(img_dir, anno_dir, save_root_dir, train_val_ratio=0.9, padding=10, debug=False):
    """
    创建训练和测试数据集
    ====================
    
    主要的数据处理函数，完成整个数据预处理流程
    
    参数:
        img_dir: 原始图像目录
        anno_dir: 标注文件目录
        save_root_dir: 处理后数据保存目录
        train_val_ratio: 训练集比例（默认0.9）
        padding: padding参数（未使用）
        debug: 调试模式
    
    处理流程:
        1. 创建标签文件（如果不存在）
        2. 图像预处理（如果不存在）
        3. 划分训练集和测试集
        4. 生成train.txt/test.txt和train.csv/test.csv文件
    """
    # 第一步：创建标签文件
    labels_dir = os.path.join(anno_dir, "labels")
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
        make_label_txt(anno_dir, labels_dir)
        print("labels done.")
    
    # 第二步：图像预处理
    save_img_dir = os.path.join(os.path.join(anno_dir, "voc2012_forYolov1"), "img")
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
        img_augument(img_dir, save_img_dir, labels_dir)
    
    # 第三步：获取所有图像文件列表
    imgs_list = os.listdir(save_img_dir)
    n_trainval = len(imgs_list)
    
    # 随机打乱数据
    shuffle_id = list(range(n_trainval))
    random.shuffle(shuffle_id)
    
    # 划分训练集和测试集
    n_train = int(n_trainval*train_val_ratio)
    train_id = shuffle_id[:n_train]
    test_id = shuffle_id[n_train:]
    
    # 第四步：生成训练集文件
    traintxt = open(os.path.join(save_root_dir, "train.txt"), 'w')
    traincsv = np.zeros((n_train, GL_NUMGRID*GL_NUMGRID*(5*GL_NUMBBOX+len(GL_CLASSES))),dtype=np.float32)
    
    for i,id in enumerate(train_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        traintxt.write(img_path)  # 写入图像路径
        
        # 读取对应的标签文件并转换为训练格式
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            traincsv[i,:] = convert_bbox2labels(bbox)  # 转换为网格格式
    
    # 保存训练集标签
    np.savetxt(os.path.join(save_root_dir, "train.csv"), traincsv)
    print("Create %d train data." % (n_train))

    # 第五步：生成测试集文件
    testtxt = open(os.path.join(save_root_dir, "test.txt"), 'w')
    testcsv = np.zeros((n_trainval - n_train, GL_NUMGRID*GL_NUMGRID*(5*GL_NUMBBOX+len(GL_CLASSES))),dtype=np.float32)
    
    for i,id in enumerate(test_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        testtxt.write(img_path)  # 写入图像路径
        
        # 读取对应的标签文件并转换为训练格式
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            testcsv[i,:] = convert_bbox2labels(bbox)  # 转换为网格格式
    
    # 保存测试集标签
    np.savetxt(os.path.join(save_root_dir, "test.csv"), testcsv)
    print("Create %d test data." % (n_trainval-n_train))


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 设置随机种子，确保结果可复现
    random.seed(0)
    np.set_printoptions(threshold=np.inf)
    
    # 设置路径
    img_dir = os.path.join(STATIC_DATASET_PATH, "JPEGImages")  # 原始图像文件夹
    anno_dirs = [STATIC_DATASET_PATH]  # 标注文件目录
    save_dir = os.path.join(STATIC_DATASET_PATH, "voc2012_forYolov1")  # 处理后数据保存目录
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # 处理每个标注目录
    for anno_dir in anno_dirs:
        create_csv_txt(img_dir, anno_dir, save_dir, debug=False)
