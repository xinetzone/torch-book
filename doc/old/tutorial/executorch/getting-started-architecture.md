# ExecuTorch 技术架构

```{topic} 背景
为了针对具有多样化硬件、关键功耗要求和实时处理需求的设备上 AI，单一的整体解决方案是不现实的。相反，需要一种模块化、分层和可扩展的架构。ExecuTorch 定义了一种简化的工作流程，用于准备（导出、转换和编译）和执行 PyTorch 程序，其中包含预定义的默认组件和明确的自定义入口点。这种架构大大提高了可移植性，使工程师能够使用高性能轻量级、跨平台的运行时，轻松集成到不同的设备和平台中。
```

将PyTorch模型部署到设备上的流程包括三个阶段：程序准备、运行时准备和程序执行，其中包含多个用户入口点。

## 程序准备

ExecuTorch 扩展了 PyTorch 在边缘设备上的灵活性和可用性。它利用 PyTorch 2 编译器和导出功能（[TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html)、[AOTAutograd](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html)、[量化](https://pytorch.org/docs/main/quantization.html)、动态形状、控制流等）来准备要在设备上执行的 PyTorch 程序。

程序准备通常简称为 AOT（提前编译），因为在最终使用 ExecuTorch 运行时（用 C++ 编写）运行程序之前，会对其进行导出、转换和编译。为了实现轻量级的运行时和低开销的执行，我们将尽可能多的工作推到 AOT 阶段。

从程序源代码开始，下面是完成程序准备所需的步骤。

### 程序源代码

- 与所有 PyTorch 用例一样，ExecuTorch 从模型创作开始，其中创建标准的 `nn.Module`` eager 模式的 PyTorch 程序。
- 导出特定的辅助工具用于表示高级功能，如[控制流](https://github.com/pytorch/executorch/blob/main/docs/website/docs/ir_spec/control_flow.md)（例如，跟踪 if-else 两个分支的辅助函数）和[动态形状](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever)（例如，数据相关的动态形状约束）。

### 导出

为了将程序部署到设备上，工程师需要使用 graph 表示来编译模型以在各种后端上运行。通过 {func}`torch.export`，生成了一个使用 ATen 方言的 EXIR（导出中间表示）。所有 AOT 编译都基于这个 EXIR，但在 lower 路径上可以有多个方言，具体细节如下。

- [ATen 方言](https://pytorch.org/executorch/stable/ir-exir.html#aten-dialect)。PyTorch Edge 基于 PyTorch 的张量库 ATen，它具有高效的执行契约。ATen 方言是由完全符合 ATen 规范的 ATe n节点表示的 graph。允许自定义算子，但必须向调度器注册。它是扁平化的，没有模块层次结构（大模块中的子模块），但源代码和模块层次结构保留在元数据中。这种表示也是自动求导安全的。

- 选地，可以在转换为 Core ATen 之前对整个 ATen 图进行量化，无论是 QAT（量化感知训练）还是 PTQ（后训练量化）。量化有助于减小模型大小，这对边缘设备非常重要。

- [Core ATen 方言](https://pytorch.org/executorch/stable/ir-ops-set-definition.html)。ATen 有数千个算子。对于一些基本的变换和内核库实现来说并不理想。从 ATen 方言图中的算子被分解为基本算子，以便算子集更小，可以应用更多的基本转换。Core ATen 方言也是可序列化的，并且可以转换为 Edge 方言，具体细节如下。

### 边缘编译

上述讨论的导出过程在最终执行代码的边缘设备上不可知的 graph 上进行运算。在边缘编译步骤中，我们使用特定于边缘的表示形式进行工作。

- [Edge 方言](https://pytorch.org/executorch/stable/ir-exir.html#edge-dialect)。所有算子都符合具有 dtype 和内存布局信息的 ATen 算子（表示为 `dim_order`），或者是已注册的自定义算子。标量被转换为张量。这些规范允许后续步骤专注于较小的边缘领域。此外，它还支持基于特定数据类型和内存布局的选择性构建。

使用 Edge 方言，有两种目标感知的方法可以进一步将 graph 降低到[后端方言](https://pytorch.org/executorch/stable/compiler-backend-dialect.html)。在这一点上，特定硬件的委托可以执行许多操作。例如，iOS上 的 Core ML、高通的 QNN 或 Arm 的 TOSA 可以重写 graph。此级别的选项包括：

- [后端委托](https://pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)。将 graph（完整或部分）编译为特定后端的入口点。在此转换过程中，用语义等效的图替换已编译的图。稍后在运行时将已编译的图卸载到后端（即委托）以提高性能。
- 用户定义的通道。还可以由用户执行目标特定的转换。这方面很好的例子是内核融合、异步行为、内存布局转换等。

### 编译为 ExecuTorch 程序

上述边缘程序适合编译，但不适合运行时环境。设备部署工程师可以降低 graph 的复杂性，以便运行时能够高效加载和执行。

在大多数边缘环境中，动态内存分配/释放会带来显著的性能和功耗开销。可以使用 AOT 内存规划和静态执行图来避免这种情况。

ExecuTorch 运行时是静态的（在图形表示方面如此，但仍然支持控制流和动态形状）。为了避免输出创建和返回，所有功能算子表示都转换为 out 变体（将输出作为参数传递）。

可选地，用户可以应用自己的内存规划算法。例如，嵌入式系统可以有特定的内存层次结构层。用户可以针对该内存层次结构进行自定义的内存规划。

程序以 ExecuTorch 运行时可以识别的格式发出。

最后，可以将发出的程序序列化为 [flatbuffer](https://github.com/pytorch/executorch/blob/main/schema/program.fbs) 格式。

## 运行时准备

使用序列化后的程序，并提供内核库（用于算子调用）或后端库（用于委托调用），模型部署工程师现在可以为运行时准备程序。

ExecuTorch 具有选择性构建 API，可以构建仅链接到程序使用的内核的运行时，从而在生成的应用程序中提供显著的二进制大小节省。

## 程序执行
ExecuTorch运行时是用C++编写的，具有最小的依赖项，以实现可移植性和执行效率。由于程序经过良好的AOT准备，核心运行时组件非常少，包括：

- 平台抽象层
- 日志记录和可选的性能分析
- 执行数据类型
- 内核和后端注册表
- 内存管理

Executor 是加载和执行程序的入口点。从这个非常小的运行时触发相应的算子内核或后端执行。

## SDK

对于用户来说，使用上述流程从研究到生产应该是高效的。生产力对于用户来说非常重要，以便他们能够创作、优化和部署他们的模型。提供 [ExecuTorch SDK](https://pytorch.org/executorch/stable/sdk-overview.html) 来提高生产力。SDK 是一个工具集，涵盖了开发者在三个阶段中的工作流程。

在程序准备和执行期间，用户可以使用 ExecuTorch SDK 对程序进行性能分析、调试或可视化。由于端到端的流程位于 PyTorch 生态系统中，用户可以关联并显示性能数据以及图形可视化，同时还可以引用程序源代码和模型层次结构。我们认为这对于快速迭代并将 PyTorch 程序降低到边缘设备和环境中是一个关键组件。
