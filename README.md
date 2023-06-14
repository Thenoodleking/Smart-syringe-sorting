# Smart-syringe-sorting

# 项目名称：
基于无监督缺陷检测模型的医用注射器自动分拣系统

# 项目目的：
在医用注射器的实际生产过程中，难免面会出现划痕、墨圈、斑点等表面缺陷。然而，传统的人工分拣方法存在检测成本高、效率低和误检率高等问题。本课题基于深度无监督学习算法构造缺陷检测模型，并配合机械臂完成医用注射器异常样本的自动化分拣。该方法能够高效、准确的分拣出异常样品，同时降低缺陷检测成本，进一步提升企业生产效益。

# 简单描述：
使用OpenCV将采集到的图片经过去噪、旋转、裁剪等操作，使用VGGNet和SegNet卷积神经网络构建去噪卷积神经网络模型。训练好后将训练好的模型迁移到树莓派（ROS、Ubuntu18.04LTS系统）中运行，对比采集到的图像与根据模型重构的图像，识别有缺陷的部分，随后调用机械臂将有缺陷的注射器分拣出来。

# 安装说明：
## 硬件说明：
树莓派4B、树莓派扩展版、ArmPi FPV机械臂（LX-15D舵机、LX-225舵机、LX-15D舵机、合金爪）、120°广角摄像头、小型传送带（绿色）
## 软件说明：
ROS操作系统、Ubuntu18.04LTS（改）、Python3.9
## 使用到的python库：
torch、cv2、numpy、Image、ImageEnhance、os、time、argparse、math、matplotlab、random、transforms、shutil、datetime、torchvision、pytorch_ssim、cae_based_segnet、operator、heapq、pickle、pandas、sklearn、data_loader_noise
# 注意
在这个项目中，由于使用了GitHub桌面版，不能上传超过100Mb的文件，所以没有上传训练好的模型，需要自行训练，其中提供参数如下：


# 使用方法：
详细的说明如何使用这个项目，包括如何启动、如何运行、如何调试等。


# 项目结构和组成部分：
描述项目的整体结构以及它的各个组成部分，这有助于其他人更好地理解项目并快速定位问题。
