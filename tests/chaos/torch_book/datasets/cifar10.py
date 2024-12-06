"""Cifar10"""
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from ..transforms.cutout import Cutout


@dataclass
class Cifar10:
    """Download the Cifar10 dataset and then load it into memory."""
    root: str = "../data"
    batch_size: int = 64
    num_workers: int = 1
    cutout: Cutout|None = Cutout(1, 16) # 数据做 Cutout 增强

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
        _train_form = [
            transforms.RandomCrop(36, padding=4),  # 先四周填充0，在吧图像随机裁剪成36*36
            # 随机裁剪出一个高度和宽度均为 upsample 像素的正方形图像，
            # 生成一个面积为原始图像面积 0.64 到 1 倍的小正方形，
            # 然后将其缩放为高度和宽度均为 32 像素的正方形
            transforms.RandomResizedCrop(32,
                                         scale=(0.64, 1.0),
                                         ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(), *_trans]
        if self.cutout is not None:
            _train_form += [self.cutout]
        self.train_form = transforms.Compose(_train_form)

    def train(self):
        return CIFAR10(root=self.root, train=True,
                                transform=self.train_form, download=True)

    def val(self):
        return CIFAR10(root=self.root, train=False,
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
