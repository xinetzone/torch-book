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
        grid = grid.as_subclass(cls)
        return grid
    
    def flatten(self) -> torch.Tensor:
        """将 Grid 数据展平为 (N, C, H, W) 形状的 Tensor"""
        num_rows, num_cols, c, h, w = self.shape
        data = self.reshape(num_rows*num_cols, c, h, w)
        return tv_tensors.wrap(data, like=self)

class PairedGrid(torch.nn.Module):
    """将 LR 和 HR 图像裁剪成 Grid 数据对"""
    def __init__(self, scale: int, config: GridConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.config = config

    def forward(self, lr: torch.Tensor, hr: torch.Tensor):
        hr_gird = Grid(hr, config=self.config)
        lr_gird = Grid(lr, config=self.config>>self.scale)
        return lr_gird, hr_gird

class FlattenPairedGrid(torch.nn.Module):
    """将 LR 和 HR 图像裁剪成 Grid 数据对展平"""
    def forward(self, lr: Grid, hr: Grid) -> torch.Tensor:
        return lr.flatten(), hr.flatten()

class PairedRandomCrop(torch.nn.Module):
    """一种用于图像数据增强的技术，通常用于生成图像对（例如高分辨率图像和低分辨率图像）的训练数据。
    
    主要目的是确保在数据增强过程中，高分辨率图像和低分辨率图像的裁剪区域保持一致，从而保证训练数据的配对关系。
    """
    def __init__(self, scale: int, gt_patch_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gt_patch_size = gt_patch_size
        self.scale = scale
        self.lq_patch_size = self.gt_patch_size // self.scale

    def forward(self, lr: torch.Tensor, hr: torch.Tensor):
        h_lq, w_lq = lr.shape[-2:]
        h_gt, w_gt = hr.shape[-2:]
        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {self.scale}x '
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < self.lq_patch_size or w_lq < self.lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                f'({self.lq_patch_size}, {self.lq_patch_size}). Please check it.')
        
        # 随机选择图像块的左上角坐标
        top = np.random.randint(h_lq - self.lq_patch_size + 1)
        left = np.random.randint(w_lq - self.lq_patch_size + 1)
        lq = lr[..., top:top + self.lq_patch_size, left:left + self.lq_patch_size]
        # 裁剪对应的 GT（Ground Truth）块。
        top_gt, left_gt = int(top * self.scale), int(left * self.scale)
        gt = hr[..., top_gt:top_gt + self.gt_patch_size, left_gt:left_gt + self.gt_patch_size,]
        return tv_tensors.wrap(lq, like=lr), tv_tensors.wrap(gt, like=hr)