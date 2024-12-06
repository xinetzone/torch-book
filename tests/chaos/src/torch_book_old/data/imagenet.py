from pathlib import Path
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision import transforms


class Transforms:
    def __init__(self, size=224):
        self.size = size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[123.68, 116.779, 103.939],
        #                                  std=[58.393, 57.12, 57.375])

        self.train = transforms.Compose([
            transforms.RandomResizedCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            normalize,
        ])


class ImageNet:
    def __init__(self, root, size=224):
        self.root = Path(root)
        self.transforms = Transforms(size)

    @property
    def trainset(self):
        return ImageFolder(self.root/"train", self.transforms.train)

    @property
    def testset(self):
        return ImageFolder(self.root/"val", self.transforms.test)

    def split(self, dtype, batch_size):
        if dtype == "train":
            dataset = self.trainset
            sampler = RandomSampler(dataset)
        else:
            dataset = self.testset
            sampler = SequentialSampler(dataset)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler)


if __name__ == "__main__":
    root = "/media/pc/data/4tb/lxw/tests/datasets/ILSVRC"
    dataset = ImageNet(root)
    trainset = dataset.split("train", batch_size=30)
    valset = dataset.split("val", batch_size=50)
