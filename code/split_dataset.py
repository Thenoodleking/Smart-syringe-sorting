import os
import random
import shutil

# 数据集中各个目录的名称及比例
dirs = ['train', 'valid', 'test']
percentages = [0.6, 0.2, 0.2]

# 定义原始图片所在文件夹和存放目录
src_dir = './dataset/syringe'
dst_root_dir = './dataset'

# 遍历文件夹中的图片，并将其划分到不同的目录中
for root, _, files in os.walk(src_dir):
    # 创建存放目录
    for d in dirs:
        os.makedirs(os.path.join(dst_root_dir, d), exist_ok=True)

    # 随机打乱文件列表
    random.shuffle(files)

    # 根据比例划分到不同的目录
    count = len(files)
    train_num = int(count * percentages[0])
    valid_num = int(count * percentages[1])
    test_num = count - train_num - valid_num

    for i, f in enumerate(files):
        if i < train_num:
            dst_dir = 'train'
        elif i < (train_num + valid_num):
            dst_dir = 'valid'
        else:
            dst_dir = 'test'

        src_path = os.path.join(root, f)
        dst_path = os.path.join(dst_root_dir, dst_dir, f)
        shutil.copy(src_path, dst_path)