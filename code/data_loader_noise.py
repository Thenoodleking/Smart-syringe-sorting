import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0):

        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        img = np.array(img,dtype= np.float32)
        #第二个参数为标准差 variance为方差
        noise = np.random.normal(self.mean, self.variance ** 0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        return torch.from_numpy(out).type(torch.FloatTensor)


class TrainDataset(Dataset):

    def __init__(self, root, var):
        # 所有图片的绝对路径
        self.var = var
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        imgs.sort()


        self.transform = transforms.Compose(
            [transforms.Resize((240,320)),
             transforms.ToTensor()]
        )
        self.noise_transform = transforms.Compose(
            [
                AddGaussianNoise(mean=0.0,variance=self.var),
            ]
        )
    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transform(pil_img)
        noise_data = self.noise_transform(data)
        return data, noise_data

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):

    def __init__(self, root, label_root, var):
        # 所有图片的绝对路径
        self.var = var
        self.root = root
        self.label_root = label_root
        imgs = os.listdir(root)
        label_imgs = os.listdir(label_root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.label_imgs = [os.path.join(label_root,k) for k in label_imgs]
        imgs.sort()
        label_imgs.sort()

        self.transform = transforms.Compose(
            [transforms.Resize((240,320)),
             transforms.ToTensor()]
        )
        self.noise_transform = transforms.Compose(
            [
                AddGaussianNoise(mean=0.0,variance=self.var),
            ]
        )
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_img_path = self.label_imgs[index]
        pil_img = Image.open(img_path)
        label_pic_img = Image.open(label_img_path)
        data = self.transform(pil_img)
        label_data = self.transform(label_pic_img)
        noise_data = self.noise_transform(data)
        return data, noise_data, label_data

    def __len__(self):
        return len(self.imgs)

    def get_data_path(self):
        return self.root

    def get_label_path(self):
        return self.label_root



def train_loader(train_path, var, batch_size=16, num_workers=4, pin_memory=True, shuffle=True):
    return DataLoader(dataset= TrainDataset(train_path, var),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)


def valid_loader(valid_path, var, batch_size=16, num_workers=4, pin_memory=True, shuffle=False):
    return DataLoader(dataset=TrainDataset(valid_path, var),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)

def test_loader(test_path, label_path,  var, batch_size=16, num_workers=4, pin_memory=True, shuffle=False):
    return DataLoader(dataset=TestDataset(test_path,label_path, var),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)