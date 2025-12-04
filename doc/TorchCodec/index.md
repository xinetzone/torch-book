# TorchCodec 教程

TorchCodec 是一个 Python 库，用于在 CPU 和 CUDA GPU 上将视频和音频数据解码为 PyTorch 张量。它的目标是快速、易于使用，并很好地集成到 PyTorch 生态系统中。如果您想使用 PyTorch 在视频和音频上训练 ML 模型，TorchCodec 就是将这些模型转换为数据的方式：
- 镜像 Python 和 PyTorch 约定的 Pythonic API。
- 依靠 [FFmpeg](https://www.ffmpeg.org/) 进行解码/编码。TorchCodec 使用您已经安装的 FFmpeg 版本。FMPEG 是一个成熟的库，在大多数系统上都具有广泛的覆盖范围。然而，它并不容易使用。TorchCodec 抽象了 FFmpeg 的复杂性，以确保正确有效地使用它。
- 将数据作为 PyTorch 张量返回，随时可以输入到 PyTorch 变换中或直接用于训练模型。

```{toctree}
install
audio/index
video/index
```