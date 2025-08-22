# 分发器

参考：[what-and-why-is-torch-dispatch](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557/1)

分发器（Dispatcher）在 PyTorch 中起着非常重要的作用，它是 PyTorch 框架实现算子动态分派（Dynamic Dispatch）的核心机制之一。它的主要作用是根据输入的类型（如张量的设备类型、数据类型等）决定具体调用哪个内核（Kernel）来执行算子。以下是分发器在 PyTorch 中的具体作用：
1. 动态分派（Dynamic Dispatch）
    - PyTorch 的算子（如加法、矩阵乘法等）是通用的，它们可以作用于不同设备（CPU 或 GPU）以及不同数据类型的张量。
    - 分发器根据算子的输入参数（如张量的设备、数据类型等）动态选择最合适的实现代码（即具体的内核函数）来执行。
2. 支持多重后端
    - PyTorch 支持多个后端（如 CPU、CUDA、XLA 等），而分发器的作用是根据当前张量所在的设备，决定将操作分发到哪一个后端进行计算。
3. 提高性能
    - 通过分发器，PyTorch 能够将操作高效地映射到最适合的硬件加速实现上，从而最大化计算性能。
4. 简化接口设计
    - 对用户来说，操作符的调用方式是统一的，而无需关心底层如何根据输入参数选择实现。这种抽象使得接口更简洁，用户无需关注底层细节。
5. 支持自定义扩展
    - 开发者可以通过 PyTorch 的扩展机制注册新的后端实现，分发器会自动识别这些扩展并根据需要进行调用。
举个例子：
当你执行 `torch.add(a, b)` 时：
    - 分发器会检查 `a` 和 `b` 的设备类型（CPU 或 CUDA）、数据类型（如 `float32`、`int64`）等信息。
    - 然后它会决定是调用 CPU 的加法实现，还是 CUDA 的加法实现，或者其他自定义的实现。
总之，分发器是 PyTorch 实现高效、灵活、跨平台计算的关键组件之一。

```{toctree}
intro
extending-torch-python-api/index
TorchDispatchMode
FlopCounterMode
```
