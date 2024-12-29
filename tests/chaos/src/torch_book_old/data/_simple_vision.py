import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ..utils import DataModule
from ..plotx import show_images

_numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)

class SimpleVison(DataModule):
    def __init__(self, batch_size=64, resize=(32, 32)):
        super().__init__()
        self.save_hyperparameters()
        self.labels = [] # 需要重写
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = self._train(trans)
        self.val = self._val(trans)
        
    def _train(self, transform):
        NotImplemented

    def _val(self, transform):
        NotImplemented

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train,
                          num_workers=self.num_workers)

    def text_labels(self, indices):
        """Return text labels."""
        return [self.labels[int(i)] for i in indices]

class FashionMNIST(SimpleVison):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    def _train(self, transform):
        return datasets.FashionMNIST(root=self.root, train=True,
                                     transform=transform, download=True)

    def _val(self, transform):
        return datasets.FashionMNIST(root=self.root, train=False,
                                     transform=transform, download=True)
        
    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)


class Cifar10(SimpleVison):
    """Download the Cifar10 dataset and then load it into memory."""

    def __init__(self, batch_size=64, resize=(32, 32)):
        super().__init__()
        self.save_hyperparameters()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.labels = ['airplane','automobile',
                       'bird', 'cat',
                       'deer', 'dog',
                       'frog', 'horse',
                       'ship', 'truck']
        normal_trans = [transforms.ToTensor(),
                        # 标准化(ImageNet)图像的每个通道
                        transforms.Normalize(self.mean, self.std)]
        trans = transforms.Compose([  # 在高度和宽度上将图像放大到40像素的正方形
            transforms.Resize(40),
            # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
            # 生成一个面积为原始图像面积0.64到1倍的小正方形，
            # 然后将其缩放为高度和宽度均为32像素的正方形
            transforms.RandomResizedCrop(32,
                                         scale=(0.64, 1.0),
                                         ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            *normal_trans])
        test_sform = transforms.Compose(normal_trans)
        self.train = self._train(trans)
        self.val = self._val(test_sform)

    def _train(self, transform):
        return datasets.CIFAR10(root=self.root, train=True,
                                transform=transform, download=True)

    def _val(self, transform):
        return datasets.CIFAR10(root=self.root, train=False,
                                transform=transform, download=True)
        
    def to_images(self, inputs, channel_layout="first"):
        inputs = _numpy(inputs)
        if channel_layout == 'first':
            inputs = inputs.transpose(0, 2, 3, 1)
        mean, std = (np.array(x) for x in [self.mean, self.std])
        inputs = std * inputs + mean
        inputs = np.clip(inputs, 0, 1)
        return inputs
    
    def visualize(self, batch, nrows=1, ncols=8, 
                  channel_layout="first", labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        imgs = self.to_images(X, channel_layout=channel_layout)
        show_images(imgs, nrows, ncols, titles=labels)