from pathlib import Path
from typing import Any
from dataclasses import dataclass
from zipfile import ZipFile
import io
from PIL import Image
import numpy as np
from ...data.cv.zipfile import LoadBufferFromZipFile

@dataclass
class PairedDIV2K:
    """成对图片数据集"""
    scale: int # 放大倍数, 2, 3, 4
    HR_path: str | Path # HR zip 数据路径
    LR_path: str | Path  # LR zip 数据路径
    

    def __post_init__(self):
        self.hr_dataset = LoadBufferFromZipFile(self.HR_path)
        self.lr_dataset = LoadBufferFromZipFile(self.LR_path)
        self._check()

    def _check(self):
        """检查图片对是否匹配"""
        assert len(self.lr_dataset) == len(self.hr_dataset)
        for a, b in zip(self.lr_dataset.filenames, self.hr_dataset.filenames):
            a = Path(a)
            a = a.name.removesuffix(a.suffix)
            b = Path(b)
            b = b.name.removesuffix(b.suffix)
            assert a == f"{b}x{self.scale}", f"文件名 {a} 和 {b} 不匹配"

    def __getitem__(self, index: int) -> list[np.ndarray, np.ndarray]:
        """加载(LR, HR)图片对
        
        Args:
            index: 图片的索引
        Returns:
            buffer: 图片的二进制内容
        """
        with Image.open(io.BytesIO(self.lr_dataset[index])) as im:
            lr = np.asanyarray(im)
        with Image.open(io.BytesIO(self.hr_dataset[index])) as im:
            hr = np.asanyarray(im)
        return lr, hr
    
    def __len__(self):
        """返回图片对数量"""
        return len(self.hr_dataset)
