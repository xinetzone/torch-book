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


def load_data_cifar10(batch_size, resize=None, num_workers=4):
    """Download the Cifar10 dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    _train = datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    _test = datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)
    return (DataLoader(_train, batch_size, shuffle=True,
                       num_workers=num_workers),
            DataLoader(_test, batch_size, shuffle=False,
                       num_workers=num_workers))
