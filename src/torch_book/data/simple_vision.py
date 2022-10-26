from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_data_fashion_mnist(batch_size, resize=None, num_workers=4):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(mnist_test, batch_size, shuffle=False,
                       num_workers=num_workers))


def load_data_cifar10(batch_size, image_size=32, num_workers=4):
    """Download the Cifar10 dataset and then load it into memory."""
    IMG_SIZE=image_size
    transform_train = transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        transforms.RandomResizedCrop(32, 
                                     scale=(0.64, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 标准化图像的每个通道
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                            [0.2023, 0.1994, 0.2010])])
    # 数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                            [0.2023, 0.1994, 0.2010])])
    # trans = [transforms.ToTensor()]
    # if resize:
    #     trans.insert(0, transforms.Resize(resize))
    # trans = transforms.Compose(trans)
    _train = datasets.CIFAR10(
        root="../data", train=True, transform=transform_train, download=True)
    _test = datasets.CIFAR10(
        root="../data", train=False, transform=transform_test, download=True)
    return (DataLoader(_train, batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(_test, batch_size, shuffle=False,
                       num_workers=num_workers))
