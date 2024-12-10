from typing import Any
from dataclasses import dataclass
import numpy as np
import torch
from torchvision import tv_tensors

@dataclass
class GridConfig:
    """网格的配置"""
    crop_size: int = 480 # HR 的裁剪尺寸，这个尺寸通常是预先设定的
    step: int = 240 # 指在 HR 图像进行某种处理时，每次移动或采样的步长
    thresh_size: int = 0 # HR 图像处理中，用于判断或筛选某些特征或对象的尺寸阈值

    def make_space(self, h, w)->list[int, int]:
        h_space = np.arange(0, h - self.crop_size + 1, self.step)
        if h - (h_space[-1] + self.crop_size) > self.thresh_size:
            h_space = np.append(h_space, h - self.crop_size)
        w_space = np.arange(0, w - self.crop_size + 1, self.step)
        if w - (w_space[-1] + self.crop_size) > self.thresh_size:
            w_space = np.append(w_space, w - self.crop_size)
        return h_space, w_space

    def __rshift__(self, scale: int):
        """将网格配置按 `scale` 因子缩小"""
        return GridConfig(
            crop_size = self.crop_size//scale, # LR 的裁剪尺寸，这个尺寸通常是预先设定的
            step = self.step//scale, # 指在 LR 图像进行某种处理时，每次移动或采样的步长
            thresh_size = self.thresh_size//scale, # LR 图像处理中，用于判断或筛选某些特征或对象的尺寸阈值
        )

class Grid(tv_tensors.TVTensor):
    def __new__(
        cls,
        data: Any,
        *,
        config: GridConfig = GridConfig(),
    ) -> "Grid":
        C, H, W = data.shape
        h_space, w_space = config.make_space(H, W)
        height = len(h_space)
        width = len(w_space)
        shape = (height, width, C, config.crop_size, config.crop_size)
        grid = torch.full(
            shape, fill_value=0,
            dtype=data.dtype, 
            device=data.device, 
            requires_grad=data.requires_grad
        )
        for row, y in enumerate(h_space):
            for col, x in enumerate(w_space):
                grid[row, col] = data[:, y:y + config.crop_size, x:x + config.crop_size]
        grid = grid.reshape(height*width, C, config.crop_size, config.crop_size)
        grid = grid.as_subclass(cls)
        grid.height = height
        grid.width = width
        return grid
