import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import os
import time
import argparse
from torch.nn import init
import math



class CAE(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(CAE, self).__init__()

        batchNorm_momentum = 0.1
        self.weights_new = self.state_dict()


        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re11 = nn.ReLU()
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re12 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re21 = nn.ReLU()
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re22 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re31 = nn.ReLU()
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re32 = nn.ReLU()
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re33 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re41 = nn.ReLU()
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re42 = nn.ReLU()
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re43 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re51 = nn.ReLU()
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re52 = nn.ReLU()
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re53 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)


        self.pool5d = nn.MaxUnpool2d(2, 2)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re53d = nn.ReLU()
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re52d = nn.ReLU()
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re51d = nn.ReLU()


        self.pool4d = nn.MaxUnpool2d(2, 2)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re43d = nn.ReLU()
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re42d = nn.ReLU()
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re41d = nn.ReLU()

        self.pool3d = nn.MaxUnpool2d(2, 2)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re33d = nn.ReLU()
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re32d = nn.ReLU()
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re31d = nn.ReLU()

        self.pool2d = nn.MaxUnpool2d(2, 2)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re22d = nn.ReLU()
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re21d = nn.ReLU()

        self.pool1d = nn.MaxUnpool2d(2, 2)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re12d = nn.ReLU()
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(label_nbr, momentum=batchNorm_momentum)
        self.re11d = nn.ReLU()


    def forward(self, x):
        # Stage 1
        x11 = self.re11(self.bn11(self.conv11(x)))
        x12 = self.re12(self.bn12(self.conv12(x11)))
        x1p, id1 = self.pool1(x12)


        # Stage 2
        x21 = self.re21(self.bn21(self.conv21(x1p)))
        x22 = self.re22(self.bn22(self.conv22(x21)))
        x2p, id2 = self.pool1(x22)

        # Stage 3
        x31 = self.re31(self.bn31(self.conv31(x2p)))
        x32 = self.re32(self.bn32(self.conv32(x31)))
        x33 = self.re33(self.bn33(self.conv33(x32)))
        x3p, id3 = self.pool3(x33)

        # Stage 4
        x41 = self.re41(self.bn41(self.conv41(x3p)))
        x42 = self.re42(self.bn42(self.conv42(x41)))
        x43 = self.re43(self.bn43(self.conv43(x42)))
        x4p, id4 = self.pool4(x43)

        # Stage 5
        x51 = self.re51(self.bn51(self.conv51(x4p)))
        x52 = self.re52(self.bn52(self.conv52(x51)))
        x53 = self.re53(self.bn53(self.conv53(x52)))
        x5p, id5 = self.pool5(x53)


        # Stage 5d
        x5d = self.pool5d(x5p, id5, x53.size())
        x53d = self.re53d(self.bn53d(self.conv53d(x5d)))
        x52d = self.re52d(self.bn52d(self.conv52d(x53d)))
        x51d = self.re51d(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.pool4d(x51d, id4, x43.size())
        x43d = self.re43d(self.bn43d(self.conv43d(x4d)))
        x42d = self.re42d(self.bn42d(self.conv42d(x43d)))
        x41d = self.re41d(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.pool3d(x41d, id3, x33.size())
        x33d = self.re33d(self.bn33d(self.conv33d(x3d)))
        x32d = self.re32d(self.bn32d(self.conv32d(x33d)))
        x31d = self.re31d(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.pool2d(x31d, id2, x22.size())
        x22d = self.re22d(self.bn22d(self.conv22d(x2d)))
        x21d = self.re21d(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.pool1d(x21d, id1, x12.size())
        x12d = self.re12d(self.bn12d(self.conv12d(x1d)))
        x11d = self.re11d(self.bn11d(self.conv11d(x12d)))

        return x11d
