from PIL import Image,ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import random
import os


train_folder = './dataset/train'
train_augmentation_folder = './dataset/train_augmentation'
if not os.path.exists(train_augmentation_folder):
    os.mkdir(train_augmentation_folder)

if __name__ == '__main__':
    files = [train_folder+'/'+f for f in os.listdir(train_folder) if f.endswith('.png')]
    count = len(files)
    for i, filename in enumerate(files):
        image = Image.open(filename)  # 用PIL中的Image.open打开图像
        j = i + count*0
        j = "%04d" % j
        name = str(j)
        newname = train_augmentation_folder + '/' + name + '.png'
        image.save(newname)
        if i % 32 == 0:
            print("rotate already successful", i)
    for i, filename in enumerate(files):
        image = Image.open(filename) # 用PIL中的Image.open打开图像
        image = image.rotate(random.randint(-5,5))#随机角度旋转
        j=i+count*1
        j="%04d"%j
        name=str(j)
        newname=train_augmentation_folder+'/'+name+'.png'
        image.save(newname)
        if i%32==0:
            print("rotate already successful",i)
    for i, filename in enumerate(files):
        image = Image.open(filename) # 用PIL中的Image.open打开图像
        image = ImageEnhance.Brightness(image).enhance(random.randint(5,15)/10.0)#随机亮度调整
        j=i+count*2
        j="%04d"%j
        name=str(j)
        newname=train_augmentation_folder+'/'+name+'.png'
        image.save(newname)
        if i%32==0:
            print("brightness already successful",i)
    for i, filename in enumerate(files):
        image = Image.open(filename) # 用PIL中的Image.open打开图像
        image = ImageEnhance.Contrast(image).enhance(random.randint(5,15)/10.0)#随机对比度调整
        j=i+count*3
        j="%04d"%j
        name=str(j)
        newname=train_augmentation_folder+'/'+name+'.png'
        image.save(newname)
        if i%32==0:
            print("constrast already successful",i)
    print("all successful")