from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

@dataclass
class Cutout:
    """随机遮挡图片的若干尺寸的若干块，尺寸和块可以根据自己的需要设置。

    参考：https://github.com/uoguelph-mlrg/Cutout
    Args:

        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    n_holes: int
    length: int

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

@dataclass
class Cifar10:
    """Download the Cifar10 dataset and then load it into memory."""
    root: str = "../data"
    batch_size: int = 64
    num_workers: int = 4

    def __post_init__(self):
        # self.mean = [0.4914, 0.4822, 0.4465]
        # self.std = [0.2023, 0.1994, 0.2010]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.labels = ['airplane', 'automobile',
                       'bird', 'cat',
                       'deer', 'dog',
                       'frog', 'horse',
                       'ship', 'truck']
        _trans = [transforms.ToTensor(),
                  # 标准化(ImageNet)图像的每个通道
                  transforms.Normalize(self.mean, self.std)]

        self.test_form = transforms.Compose(_trans)
        self.train_form = transforms.Compose([
            transforms.RandomCrop(36, padding=4),  #先四周填充0，在吧图像随机裁剪成36*36
            # 随机裁剪出一个高度和宽度均为 upsample 像素的正方形图像，
            # 生成一个面积为原始图像面积 0.64 到 1 倍的小正方形，
            # 然后将其缩放为高度和宽度均为 32 像素的正方形
            transforms.RandomResizedCrop(32,
                                         scale=(0.64, 1.0),
                                         ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            #  *_trans])
            *_trans, Cutout(n_holes=1, length=16)])

    def train(self):
        return datasets.CIFAR10(root=self.root, train=True,
                                transform=self.train_form, download=True)

    def val(self):
        return datasets.CIFAR10(root=self.root, train=False,
                                transform=self.test_form, download=True)

    def train_loader(self):
        return DataLoader(self.train(), self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_loader(self):
        return DataLoader(self.val(), self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    # def to_images(self, inputs, channel_layout="first"):
    #     inputs = _numpy(inputs)
    #     if channel_layout == 'first':
    #         inputs = inputs.transpose(0, 2, 3, 1)
    #     mean, std = (np.array(x) for x in [self.mean, self.std])
    #     inputs = std * inputs + mean
    #     inputs = np.clip(inputs, 0, 1)
    #     return inputs

    # def visualize(self, batch, nrows=1, ncols=8,
    #               channel_layout="first", labels=[]):
    #     X, y = batch
    #     if not labels:
    #         labels = self.text_labels(y)
    #     imgs = self.to_images(X, channel_layout=channel_layout)
    #     show_images(imgs, nrows, ncols, titles=labels)
