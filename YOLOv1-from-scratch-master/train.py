"""
YOLOv1训练脚本
==============

这个文件实现了YOLOv1网络的完整训练流程，包括：
1. 训练循环管理
2. 验证过程
3. 模型保存
4. 日志记录
5. 学习率调度

训练流程：
1. 加载数据集和模型
2. 设置优化器和学习率
3. 执行训练循环
4. 定期验证和保存模型
5. 记录训练日志

作者：YOLOv1实现项目
"""

import os
import datetime
import time
import torch
from torch.utils.data import DataLoader

from model import MyNet
from data import MyDataset
from my_arguments import Args
from prepare_data import GL_CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import labels2bbox


class TrainInterface(object):
    """
    YOLOv1训练接口类
    ===============
    
    负责管理整个训练过程，包括：
    - 训练循环控制
    - 验证过程管理
    - 模型保存
    - 日志记录
    """

    def __init__(self, opts):
        """
        训练接口初始化
        
        参数:
            opts: 训练参数配置对象，包含所有训练相关的超参数
        """
        self.opts = opts
        print("=======================Start training.=======================")

    @staticmethod
    def __train(model, train_loader, optimizer, epoch, num_train, opts):
        """
        单轮训练函数
        ============
        
        执行一个完整的训练epoch，包括：
        1. 前向传播计算预测结果
        2. 计算损失函数
        3. 反向传播更新参数
        4. 记录训练日志
        
        参数:
            model: YOLOv1网络模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            epoch: 当前训练轮数
            num_train: 训练样本总数
            opts: 训练参数配置
        """
        # 设置模型为训练模式
        model.train()
        device = opts.GPU_id
        avg_metric = 0.  # 平均评价指标
        avg_loss = 0.  # 平均损失数值
        
        # 打开日志文件，记录训练过程
        # log_file是保存网络训练过程信息的文件，网络训练信息会以追加的形式打印在log.txt里，不会覆盖原有log文件
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
        log_file.write(localtime)
        log_file.write("\n======================training epoch %d======================\n"%epoch)
        
        # 遍历训练数据批次
        for i,(imgs, labels) in enumerate(train_loader):
            # 重塑标签数据格式：从(batch_size, 7*7*30)转换为(batch_size, 30, 7, 7)
            labels = labels.view(opts.batch_size, GL_NUMGRID, GL_NUMGRID, -1)
            labels = labels.permute(0,3,1,2)
            
            # 将数据移动到GPU（如果使用GPU）
            if opts.use_GPU:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
            # 前向传播：计算网络预测结果
            preds = model(imgs)
            
            # 计算损失函数
            loss = model.calculate_loss(labels)
            
            # 反向传播：计算梯度并更新参数
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新网络参数
            
            # 计算平均损失（用于监控训练进度）
            # metric = model.calculate_metric(preds, labels)  # 计算评价指标
            # avg_metric = (avg_metric*i+metric)/(i+1)
            avg_loss = (avg_loss*i+loss.item())/(i+1)
            
            # 根据打印频率输出训练信息
            if i % opts.print_freq == 0:
                print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss))
                log_file.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss))
                log_file.flush()  # 立即写入文件
                
        log_file.close()

    @staticmethod
    def __validate(model, val_loader, epoch, num_val, opts):

        model.eval()
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        log_file.write("======================validate epoch %d======================\n"%epoch)
        preds = None
        gts = None
        avg_metric = 0.
        with torch.no_grad():  # 加上这个可以减少在validation过程时的显存占用，提高代码的显存利用率
            for i,(imgs, labels) in enumerate(val_loader):
                if opts.use_GPU:
                    imgs = imgs.to(opts.GPU_id)
                pred = model(imgs).cpu().squeeze(dim=0).permute(1,2,0)
                pred_bbox = labels2bbox(pred)  # 将网络输出经过NMS后转换为shape为(-1, 6)的bbox
            metric = model.calculate_metric(preds, gts)
            print("Evaluation of validation result: average L2 distance = %.5f"%(metric))
            log_file.write("Evaluation of validation result: average L2 distance = %.5f\n"%(metric))
            log_file.flush()
            log_file.close()
        return metric

    @staticmethod
    def __save_model(model, epoch, opts):

        model_name = "epoch%d.pkl" % epoch
        save_dir = os.path.join(opts.checkpoints_dir, model_name)
        torch.save(model, save_dir)


    def main(self):

        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        random_seed = opts.random_seed
        train_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="train", train_val_ratio=0.9)
        val_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="val", train_val_ratio=0.9)
        train_loader = DataLoader(train_dataset, opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        num_train = len(train_dataset)
        num_val = len(val_dataset)

        if opts.pretrain is None:
            model = MyNet()
        else:
            model = torch.load(opts.pretrain)
        if opts.use_GPU:
            model.to(opts.GPU_id)
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

        best_metric=1000000
        for e in range(opts.start_epoch, opts.epoch+1):
            t = time.time()
            self.__train(model, train_loader, optimizer, e, num_train, opts)
            t2 = time.time()
            print("Training consumes %.2f second\n" % (t2-t))
            with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t2-t))
            if e % opts.save_freq==0 or e == opts.epoch+1:
                # t = time.time()
                # metric = self.__validate(model, val_loader, e, num_val, opts)
                # t2 = time.time()
                # print("Validation consumes %.2f second\n" % (t2 - t))
                # with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                #     log_file.write("Validation consumes %.2f second\n" % (t2 - t))
                # if best_metric>metric:
                #     best_metric = metric
                #     print("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                #     with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                #         log_file.write("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                self.__save_model(model, e, opts)


if __name__ == '__main__':
    # 训练网络代码
    args = Args()
    args.set_train_args()  # 获取命令行参数
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()  # 调用训练接口
