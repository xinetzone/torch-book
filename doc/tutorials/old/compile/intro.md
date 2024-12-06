# `torch.compiler` 简介

`torch.compiler` 是一个命名空间，通过它，一些内部编译器方法可以供用户使用。该命名空间的主要功能和特点是 {func}`torch.compile`。

`torch.compile` 是 PyTorch 2.x 中引入的 PyTorch 函数，旨在解决 PyTorch 中的精确 graph 捕获问题，并最终使软件工程师能够更快地运行他们的 PyTorch 程序。{func}`torch.compile` 是用 Python 编写的，标志着 PyTorch 从 C++ 过渡到 Python。

{func}`torch.compile` 利用了以下底层技术：

- `TorchDynamo`（{mod}`torch._dynamo`）是内部 API，它使用 CPython 的 Frame Evaluation API 特性来安全地捕获 PyTorch graph。对于 PyTorch 用户来说，可以通过 `torch.compiler` 命名空间访问这些外部可用的方法。
- `TorchInductor`是 `torch.compile` 的默认深度学习编译器，它可以为多个加速器和后端生成快速的代码。要通过 {func}`torch.compile` 实现加速，需要使用后端编译器。对于 NVIDIA 和 AMD GPU，它利用 OpenAI Triton 作为关键构建块。
- AOT Autograd 不仅捕获用户级别的代码，还捕获反向传播，从而实现“提前捕获”反向传递。这使得可以使用 `TorchInductor` 加速前向和后向传递。

如上所述，要更快地运行您的工作流程，通过 `TorchDynamo` 的 `torch.compile` 需要一个后端，该后端将捕获的图转换为快速的机器代码。不同的后端可以导致不同的优化收益。默认的后端称为 `TorchInductor`，也称为 `inductor`。`TorchDynamo` 有一个由合作伙伴开发的受支持的后端列表，可以通过运行 {func}`torch.compiler.list_backends` 来查看每个后端及其可选依赖项。

## `torch.compiler` 后端

`torch.compile(m, backend=...)` 后端支持情况：
- 支持训练和推理
    - `"inductor"`：`TorchInductor`
    - `"cudagraphs"`：带有 AOT Autograd 的 CUDA graphs
    - `"ipex"`：使用 IPEX 的 CPU
    - `"onnxrt"`：使用 ONNX Runtime 在 CPU/GPU 上进行训练
- 仅支持推理
    - `"tensorrt"`：使用 ONNX Runtime 运行 TensorRT 进行推理优化。
    - `"tvm"`: 使用 Apache TVM 进行推理优化。
