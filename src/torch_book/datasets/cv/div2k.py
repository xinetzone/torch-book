from pathlib import Path
from typing import Any
from dataclasses import dataclass
from zipfile import ZipFile
import io
from PIL import Image
import numpy as np

@dataclass
class LoadDIV2KBufferFromZipFile:
    """从 `.zip` 文件中加载图片 buffer 列表"""
    path: str|Path # 数据路径

    def __post_init__(self):
        with ZipFile(self.path) as fp:
            # 获取图片名称列表，并排序
            self.filenames = sorted([file.filename for file in fp.filelist if not file.is_dir()])

    def __len__(self):
        """返回图片数量"""
        return len(self.filenames)

    def __call__(self, file_name: str) -> bytes:
        """加载图片的二进制内容

        Args:
            file_name: zip 中图片的名称，例如：'0.jpg'
        Returns:
            buffer: 图片的二进制内容
        """
        with ZipFile(self.path) as fp:
            buffer = fp.read(file_name)
        return buffer

    def __getitem__(self, index: int) -> Any:
        """加载图片的二进制内容
        Args:
            index: 图片的索引
        Returns:
            buffer: 图片的二进制内容
        """
        return self(self.filenames[index])

@dataclass
class PairedDIV2K:
    """成对图片数据集"""
    scale: int # 放大倍数, 2, 3, 4
    HR_path: str | Path # HR zip 数据路径
    LR_path: str | Path  # LR zip 数据路径
    

    def __post_init__(self):
        self.hr_dataset = LoadDIV2KBufferFromZipFile(self.HR_path)
        self.lr_dataset = LoadDIV2KBufferFromZipFile(self.LR_path)
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
