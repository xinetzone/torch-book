import io
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from zipfile import ZipFile
from PIL import Image
import torch

@dataclass
class LoadBufferFromZipFile:
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
class DIV2K(LoadBufferFromZipFile):
    transform: Any # 图片预处理操作

    def __call__(self, file_name: str) -> torch.Tensor:
        """加载图片，并转换为 torch.Tensor 对象

        Args:
            file_name: zip 中图片的名称，例如：'0.jpg'
        """
        with ZipFile(self.path) as fp:
            buffer = fp.read(file_name)
        # with Image.open(io.BytesIO(buffer)) as im:
        data = Image.open(io.BytesIO(buffer)) 
        if self.transform:
            data = self.transform(data)
        return data
