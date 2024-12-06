from torch import nn

class SRCNNNet(nn.Module):
    """SRCNN 网络结构用于图像超分辨率。

    SRCNN包含三个卷积层。对于每一层，可以定义输入通道数、输出通道数和卷积核大小。
    输入图像首先会使用双三次插值法进行上采样，然后在高分辨率空间尺寸中进行超分辨处理。

    论文：Learning a Deep Convolutional Network for Image Super-Resolution.

    Args:
        channels (tuple[int]): 元组，包含了每一层的通道数，包括输入和输出的通道数。默认值：(3, 64, 32, 3)。
        kernel_sizes (tuple[int]): 元组，包含了每个卷积层的卷积核大小。默认值：(9, 1, 5)。
        upscale_factor (int): 上采样因子。默认值：4。
    """

    def __init__(self,
                 channels=(3, 64, 32, 3),
                 kernel_sizes=(9, 1, 5),
                 upscale_factor=4,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(channels) == 4, (f'通道元组的长度应为4，但实际得到的长度是 {len(channels)}')
        assert len(kernel_sizes) == 3, f"kernel 元组的长度应为3，但得到的是{len(kernel_sizes)}"
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(
            channels[0],
            channels[1],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(
            channels[1],
            channels[2],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(
            channels[2],
            channels[3],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
