"""ImageNet"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import v2

@dataclass
class ImageNet:
    root: str

    def __post_init__(self):
        self.root = Path(self.root)
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    @property
    def trainset(self):
        return ImageFolder(self.root/"train", transform=v2.Compose([
            v2.Resize(224),
            v2.RandomCrop(224),
            v2.RandomHorizontalFlip(),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            self.normalize]))

    @property
    def testset(self):
        return ImageFolder(self.root/"val", transform=v2.Compose([
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            self.normalize]))

    def train_loader(self, batch_size):
        train_sampler = torch.utils.data.RandomSampler(self.trainset)
        return torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, sampler=train_sampler)

    def test_loader(self, batch_size):
        test_sampler = torch.utils.data.SequentialSampler(self.testset)
        return torch.utils.data.DataLoader(self.testset, batch_size=batch_size, sampler=test_sampler)
