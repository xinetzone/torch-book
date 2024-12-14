from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torchvision import tv_tensors

@dataclass
class PairedDataset(Dataset):
    """成对图片数据集"""
    scale: int # 放大倍数, 2, 3, 4
    HR_path: str | Path # HR zip 数据路径
    LR_path: str | Path  # LR zip 数据路径
    transform: Callable | None = None

    def __post_init__(self):
        self.HR_path = Path(self.HR_path)
        self.LR_path = Path(self.LR_path)
        self.lr_names = sorted(self.LR_path.iterdir())
        self.hr_names = sorted(self.HR_path.iterdir())
        self._check()
    
    def _check(self):
        """检查图片对是否匹配"""
        assert len(self.lr_names) == len(self.hr_names)
        for a, b in zip(self.lr_names, self.hr_names):
            assert Path(a).name == Path(b).name, f"文件名 {a} 和 {b} 不匹配"

    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.lr_names)

    def __getitem__(self, index: int) -> list[tv_tensors.Image, tv_tensors.Image]:
        """加载(LR, HR)图片对

        Args:
            index: 图片的索引
        Returns:
            buffer: 图片的二进制内容
        """
        with Image.open(self.lr_names[index]) as im:
            lr = tv_tensors.Image(im)
        
        with Image.open(self.hr_names[index]) as im:
            hr = tv_tensors.Image(im)
        if self.transform is not None:
            lr, hr = self.transform(lr, hr)
        return lr, hr
