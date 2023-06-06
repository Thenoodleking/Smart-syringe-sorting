import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image,make_grid
import random
import pytorch_ssim
from cae_based_segnet import *
import torch.optim as optim
from operator import itemgetter
from heapq import nsmallest
import pickle
from heapq import nlargest
from operator import itemgetter
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve, auc
import data_loader_noise
from sklearn import metrics
import time


#模型路径
model_path = "./log/new_dcae_100_epoch_ssim_adam_0.001_mean_0.0_vari_0.01_seed_0/model.pth"
#测试集路径
test_data_path = "./dataset/test"
#模型输入图像的保存路径
input_data_path = "./infer/input"
#模型输出图像的保存的路径
output_data_path = "./infer/output"


if not os.path.exists(input_data_path):
    os.makedirs(input_data_path)
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

if __name__ == '__main__':
    #加载模型
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # 遍历测试集 生成重构图像
    test_data_loader = data_loader_noise.valid_loader(test_data_path, var=0.01, batch_size=1)
    for i_batch, (raw_data, noi_data) in enumerate(test_data_loader):
        if torch.cuda.is_available():
            raw_data = raw_data.cuda()
        rec_data = model(raw_data)
        sub_data = pytorch_ssim.ssim_map(raw_data, rec_data, window_size=3)
        save_image(raw_data, input_data_path + "/" + str("%03d" % i_batch) + "_input.png", normalize=False);
        save_image(rec_data, output_data_path + "/" + str("%03d" % i_batch) + "_output.png", normalize=False);
