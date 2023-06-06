import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import random

import pytorch_ssim
import data_loader_noise
from cae_based_segnet import *
import torch.optim as optim
from operator import itemgetter
from heapq import nsmallest
# from prune_certain_filter import *
import pickle
from heapq import nlargest
from operator import itemgetter
import matplotlib.pyplot as plt


class PrunningFineTuner_CAE:
    def __init__(self, train_path, valid_path, model, var, criterion=None, optimizer=None):
        self.train_path = train_path
        self.valid_path = valid_path


        self.train_data_loader = data_loader_noise.train_loader(train_path, var, batch_size=batch_size)
        self.valid_data_loader = data_loader_noise.valid_loader(valid_path, var, batch_size=batch_size)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.model.train()

        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        if criterion is None:
            self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def compute_IoU(self, thresh_img, label_img):
        intersection = np.logical_and(thresh_img, label_img)
        union = np.logical_or(thresh_img, label_img)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def valid(self):
        self.model.eval()
        loss_sum = 0
        for i_batch, (raw_data, noi_data) in enumerate(self.valid_data_loader):
            if torch.cuda.is_available():
                raw_data = raw_data.cuda()
            rec_data = self.model(raw_data)
            loss = 1 - self.criterion(raw_data, rec_data)
            loss_sum += loss.item()
        self.model.train()
        return loss_sum / (i_batch + 1)

    def train(self, epoches=10):

        self.model.train()
        global cur_epoch

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            cur_epoch += 1

        print("Finished fine tuning.")

    def train_epoch(self):
        global train_loss_list
        global valid_loss_list

        loss_sum = 0
        for i, (raw_data, noi_data) in enumerate(self.train_data_loader):
                batch_loss = self.train_batch(i, raw_data, noi_data)
                loss_sum += batch_loss
        train_loss = loss_sum / (i + 1)
        train_loss_list.append(train_loss)


        valid_loss = self.valid()
        valid_loss_list.append(valid_loss)


        index_list.append(len(index_list))
        print("Train Loss:%.8f" % train_loss, "Valid Loss:%.8f" % valid_loss)

        # 绘图
        plt.figure(figsize=(10, 5))

        l1, = plt.plot(index_list, train_loss_list, label='Train Loss')
        l2, = plt.plot(index_list, valid_loss_list, label='Valid Loss',)
        plt.ylim(0, 0.5)
        plt.xlim(-0.1, len(index_list))
        plt.grid()
        # 常规的绘图轴、标题设置
        plt.xlabel('Epoch', fontsize=15)

        plt.ylabel('Loss', fontsize=15)

        plt.legend(fontsize=15)  # 调整图例的大小
        global log_path
        plt.savefig(log_path + "/loss_graph.png")

    def train_batch(self, i_batch, raw_data, noi_data):

        if use_cuda:
            raw_data = raw_data.cuda()
            noi_data = noi_data.cuda()

        self.optimizer.zero_grad()
        input = Variable(noi_data)

        output = self.model(input)

        #每10个Epoch存储第一个Patch的训练图像
        if i_batch == 0 and cur_epoch % 10 == 0:
            save_image(raw_data, train_img_path + '/' + str("%03d" % cur_epoch) + '_raw.png', normalize=False)
            save_image(noi_data, train_img_path + '/' + str("%03d" % cur_epoch) + '_noi.png', normalize=False)
            save_image(output, train_img_path + '/' + str("%03d" % cur_epoch) + '_rec.png', normalize=False)

        loss = 1 - self.criterion(output, Variable(raw_data))
        loss.backward()
        self.optimizer.step()
        return loss.item()




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_train_process_info():

    train_loss_path = log_path + '/train_loss.txt'
    for loss in train_loss_list:
        with open(train_loss_path, "a+") as f:
            f.write(str(loss))
            f.write('\n')

    valid_loss_path = log_path + '/valid_loss.txt'
    for loss in valid_loss_list:
        with open(valid_loss_path, "a+") as f:
            f.write(str(loss))
            f.write('\n')






#设置训练的CPU 初始化随机种子
torch.cuda.set_device(0)
setup_seed(0)
use_cuda = torch.cuda.is_available()
batch_size = 6
cur_epoch = 0

train_data_path = "./dataset/train_augmentation"
valid_data_path = "./dataset/valid"

train_loss_list = []
valid_loss_list = []

index_list = []

#训练日志、模型的保存路径
log_path = "./log/new_dcae_100_epoch_ssim_adam_0.001_mean_0.0_vari_0.01_seed_0"
if not os.path.exists(log_path):
    os.mkdir(log_path)

#训练过程添加高斯噪声的方差
var = 0.0001



if __name__ == '__main__':

    mode = 0

    #重新开始训练
    if mode == 0:

        train_img_path = log_path + "/train"
        test_img_path = log_path + "/test"
        if not os.path.exists(train_img_path):
            os.mkdir(train_img_path)
        if not os.path.exists(test_img_path):
            os.mkdir(test_img_path)


        model = CAE(3, 3)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = pytorch_ssim.SSIM()

        fine_tuner = PrunningFineTuner_CAE(train_data_path, valid_data_path,
                                          model, var, loss_func, optimizer)
        fine_tuner.train(epoches=200)
        torch.save(model,log_path+'/model.pth')
        save_train_process_info()

    #继续训练
    elif mode == 1:

        train_img_path = log_path + "/train"
        test_img_path = log_path + "/test"
        if not os.path.exists(train_img_path):
            os.mkdir(train_img_path)
        if not os.path.exists(test_img_path):
            os.mkdir(test_img_path)


        model = torch.load(log_path + '/model.pth', map_location=lambda storage, loc: storage)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = pytorch_ssim.SSIM()
        fine_tuner = PrunningFineTuner_CAE(train_data_path, valid_data_path,
                                           model, var, loss_func, optimizer)
        fine_tuner.train(200)
        torch.save(model, log_path + '/model.pth')
        save_train_process_info()


