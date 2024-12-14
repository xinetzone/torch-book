import torch

class CropBorder(torch.nn.Module):
    """裁掉图像的边界
    
    Args:
        size: 图像每条边缘裁剪的像素。这些裁剪掉的像素不参与 PSNR 的计算。默认值为 0。
    """
    def __init__(self, size: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    def forward(self, x):
        return x[..., self.size:-self.size, self.size:-self.size]