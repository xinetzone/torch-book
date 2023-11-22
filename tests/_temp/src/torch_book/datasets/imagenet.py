from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision import transforms


@dataclass
class ImageNet:
    root: str

    def __post_init__(self):
        self.root = Path(self.root)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    @property
    def trainset(self):
        return ImageFolder(self.root/"train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize]))

    @property
    def testset(self):
        return ImageFolder(self.root/"val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize]))

    def train_loader(self, batch_size):
        train_sampler = torch.utils.data.RandomSampler(self.trainset)
        return torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, sampler=train_sampler)

    def test_loader(self, batch_size):
        test_sampler = torch.utils.data.SequentialSampler(self.testset)
        return torch.utils.data.DataLoader(self.testset, batch_size=batch_size, sampler=test_sampler)
