# ExecuTorch 概念

## 提前编译

AOT（Ahead of Time）通常指在执行之前发生的程序准备。在高层次上，ExecuTorch 工作流程分为 AOT 编译和运行时。AOT 步骤涉及将代码编译成中间表示（Intermediate Representation，简称 IR），以及可选的变换和优化。

## ATen

从根本上说，它是 PyTorch 中几乎所有其他 Python 和 C++ 接口的基础张量库。它提供了核心的张量类，其中定义了数百个算子。

## ATen 方言

ATen 方言（dialect）是将即时导出的 eager 模式转换为 graph 表示的结果。它是 ExecuTorch 编译管道的入口点；在导出到 ATen 方言后，后续过程可以 lower 到 Core ATen 方言和 [Edge 方言](https://pytorch.org/executorch/stable/concepts.html#edge-dialect)。

ATen 方言是有效的 [EXIR](https://pytorch.org/executorch/stable/concepts.html#exir)，具有附加属性。它由函数式 ATen 算子、高阶算子（如控制流算子）和注册的自定义算子组成。

ATen 方言的目标是尽可能忠实地捕捉用户程序。

## ATen 模式

ATen 模式（mode）使用 PyTorch 核心中的 Tensor（`at::Tensor`）和相关类型的 ATen 实现，例如 `ScalarType`。这与可移植模式形成对比，后者使用 ExecuTorch 的较小型张量实现（`torch::executor::Tensor`）和相关类型，例如 `torch::executor::ScalarType`。

在此配置中，依赖于完整 `at::Tensor`` API 的 ATen 内核是可用的。

ATen 内核倾向于进行动态内存分配，并且通常具有额外的灵活性（以及开销），以处理不需要移动/嵌入式客户端的情况。例如，CUDA 支持、稀疏张量支持和 dtype promotion。

注意：ATen 模式目前仍在开发中。

## Autograd 安全的 ATen 方言

Autograd 安全的 ATen 方言仅包括可微分的 ATen 算子，以及高阶算子（控制流算子）和注册的自定义算子。

## 后端

特定的硬件（如 GPU、NPU）或软件堆栈（如 XNNPACK），它们消耗图形或其一部分，并具有性能和效率优势。

## 后端方言

后端方言是将 Edge 方言导出到特定后端后立即得到的结果。它是目标感知的，并且可能包含对目标后端有意义的算子或子模块。此方言允许引入与 Core ATen 算子集中定义的模式不一致且未在 ATen 或 Edge 方言中显示的目标特定算子。

## 后端注册表

将后端名称映射到后端接口的表格。这允许在运行时通过名称调用后端。

## 后端特定算子

这些算子不是 ATen 方言或 Edge 方言的一部分。后端特定算子仅由发生在 Edge 方言之后的传递引入（请参见后端方言）。这些算子特定于目标后端，并且通常执行更快。

## Buck2

一种开源的大规模构建系统，用于构建 ExecuTorch。

## CMake

一种开源的跨平台工具系列，用于构建、测试和打包软件。用于构建 ExecuTorch。

## 代码生成

在 ExecuTorch 中，代码生成用于生成[内核注册库](https://pytorch.org/executorch/stable/kernel-library-selective-build.html)。

## 核心 ATen 方言
核心 ATen 方言包含核心 ATen 算子，以及高阶算子（控制流）和已注册的自定义算子。

## 核心 ATen 算子 / 规范 ATen 算子集

PyTorch ATen 算子库的一个子集。在导出时，核心 ATen 算子不会与核心 ATen 分解表一起分解。它们作为上游期望的基本 ATen 运算的参考。

## 核心 ATen 分解表

分解算子意味着用其他算子表示它。在 AOT 过程中，使用默认的分解列表将 ATen 算子分解为核心 ATen 算子。这被称为核心 ATen 分解表。

## 自定义算子

这些不是 ATen 库的一部分，但在 eager 模式下出现。已注册的自定义算子被注册到当前的 PyTorch [eager 模式](eager-mode) 运行时中，通常通过调用 `TORCH_LIBRARY`。它们可能与特定的目标模型或硬件平台相关联。例如， [`torchvision::roi_align`](https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html) 是广泛使用的自定义算子，由 `torchvision` 使用（不针对特定硬件）。

## DataLoader

一个接口，使 ExecuTorch 运行时能够从文件或其他数据源读取，而无需直接依赖于操作系统概念，如文件或内存分配。

## 委托

在特定后端（例如 XNNPACK）上运行程序的部分（或全部），而其余程序（如果有的话）在基本的 ExecuTorch 运行时上运行。委托使我们能够利用专门后端和硬件的性能和效率优势。

## DSP

DSP（数字信号处理器）是一种专门优化了数字信号处理架构的微处理器芯片。

## dtype
数据类型，张量中的数据类型（例如 float、integer 等）。

(dynamic-quantization)=
## 动态量化

动态量化(Dynamic Quantization) 是一种量化方法，其中在推理期间即时对张量进行量化。这与静态量化形成对比，[静态量化](static-quantization)是在推理之前对张量进行量化。

## 动态形状

指模型在推理期间接受不同形状输入的能力。例如，ATen op [unique_consecutive](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html) 和自定义 op [MaskRCNN](https://pytorch.org/vision/main/models/mask_rcnn.html) 具有数据依赖的输出形状。这种算子难以进行内存规划，因为即使对于相同的输入形状，每次调用也可能产生不同的输出形状。为了在 ExecuTorch 中支持动态形状，内核可以使用客户端提供的 MemoryAllocator 分配张量数据。

(eager-mode)=
## Eager 模式

Python 执行环境，其中模型中的算子在遇到时立即执行。例如，Jupyter / Colab 笔记本在 eager 模式下运行。这与 graph 模式形成对比，在 graph 模式下，算子首先被合成为计算图，然后进行编译和执行。

## Edge 方言

一种 EXIR 的方言，具有以下属性：

- 所有算子都来自预定义的算子集，称为“边缘算子”或已注册的自定义算子。
- 计算图和每个节点的输入和输出必须是张量。所有标量类型都转换为张量。

边缘方言引入了对边缘设备有用的特化，但不一定适用于通用（服务器）导出。但是，除了原始 Python 程序中已经存在的那些之外，边缘方言不包含针对特定硬件的特化。

## 边缘算子

具有 dtype 特化的 ATen 算子。

## ExecuTorch

ExecuTorch 是一个统一的机器学习软件栈，位于 PyTorch Edge 平台内，专为高效的设备端推理而设计。ExecuTorch 定义了一个工作流程，用于在边缘设备（如移动设备、可穿戴设备和嵌入式设备）上准备（导出和转换）并执行 PyTorch 程序。

## ExecuTorch 方法

ExecuTorch 方法：相当于 `nn.Module` Python 方法的可执行版本。例如，{func}`forward` Python 方法将编译为 ExecuTorch 方法。

## ExecuTorch 程序

ExecuTorch 程序将字符串名称（如 `forward`）映射到特定的 ExecuTorch 方法条目。

## executor_runner

围绕 ExecuTorch 运行时的示例包装器，包括所有算子和后端。

(exir)=
## EXIR
从 {func}`torch.export` 导出的中间表示（IR）。包含模型的计算图。所有 EXIR 图都是有效的 [FX 图](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)。

## ExportedProgram

{func}`torch.export` 的输出，将 PyTorch 模型（通常是 `nn.Module`）的计算图与模型使用的参数或权重捆绑在一起。

## flatbuffer

内存高效的跨平台序列化库。在 ExecuTorch 的上下文中，急切模式的 Pytorch 模型被导出为 flatbuffer 格式，这是 ExecuTorch 运行时所使用的格式。

## Framework tax

各种加载和初始化任务（非推理）的成本。例如：加载程序、初始化执行器、内核和后端委托分派以及运行时内存利用。

## Functional ATen operators

没有副作用的 ATen 算子。

## Graph

EXIR 图是以 DAG（有向无环图）形式表示的 PyTorch 程序。图中的每个节点代表特定的计算或操作，而图的边缘由节点之间的引用组成。注意：所有 EXIR 图都是有效的 [FX 图](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)。

(graph-mode)=
## Graph mode

在图模式下，算子首先被综合成一个图，然后整体进行编译和执行。这与急切模式相反，在急切模式下，算子在遇到时立即执行。图模式通常提供更高的性能，因为它允许进行诸如算子融合等优化。

## 高阶算子
高阶算子（Higher Order Operators，简称 HOP）是一种接受 Python 函数作为输入、返回 Python 函数作为输出或同时具有这两种功能的算子。

与所有 PyTorch 算子一样，高阶算子也可以选择性地为后端和功能实现提供实现。这使我们能够为高阶算子注册自动梯度公式，或者定义高阶算子在 ProxyTensor 跟踪下的行为。

## Hybrid Quantization

一种量化技术，根据计算复杂度和对精度损失的敏感性，使用不同的技术对模型的不同部分进行量化。为了保持准确性，模型的某些部分可能不会被量化。

## Intermediate Representation (IR)

源语言和目标语言之间的程序表示形式。通常，它是编译器或虚拟机内部使用的数据结构，用于表示源代码。

## Kernel

算子的实现。可以为不同的后端/输入等实现多个算子的实现。

## Kernel registry / Operator registry

内核名称及其实现之间的映射表。这允许 ExecuTorch 运行时在执行期间解析对内核的引用。

## Lowering

将模型转换为在不同后端上运行的过程。由于它使代码更接近硬件，因此被称为“降低”。在 ExecuTorch 中，作为后端委托的一部分执行降低操作。

## Memory planning

为模型分配和管理内存的过程。在 ExecuTorch 中，在将图保存到 flatbuffer 之前运行内存规划过程。这会为每个张量分配一个内存 ID 和缓冲区中的偏移量，标记张量的存储位置。

## Node

EXIR 图中的节点表示特定的计算或操作，并使用 Python 中的 {class}`torch.fx.Node` 类表示。

## Operator

张量上的函数。这是抽象的概念，内核是实现。可以为不同的后端/输入等提供不同的实现。

## Operator fusion

算子融合是将多个算子合并成一个复合算子的过程，由于更少的内核启动和更少的内存读写，导致更快的计算。这是图模式相对于急切模式的性能优势之一。

## Out variant

算子的 out 变体不是在内核实现中分配返回的张量，而是将预先分配的张量作为其 out 参数传入，并将结果存储在那里。

这使得内存规划器更容易执行张量生命周期分析。在 ExecuTorch 中，在内存规划之前执行 out 变体过程。

## PAL

平台抽象层(Platform Abstraction Layer)为执行环境提供一种方式，可以覆盖以下操作：

- 获取当前时间。
- 打印日志语句。
- 使进程/系统崩溃。默认的 PAL 实现可以在特定客户端系统上进行覆盖，如果它不起作用的话。

## 部分内核

支持张量数据类型和/或维度顺序的内核。

(partitioner)=
## 分区器

分区器(Partitioner)模型的某些部分可能被委托给运行在优化后的后端。分区器将图分割成适当的子网络，并为它们标记以进行委托。

## 便携式模式（精简模式）

便携式模式(Portable mode/lean mode)使用 ExecuTorch 较小的张量实现（`torch::executor::Tensor`），以及相关的类型（`torch::executor::ScalarType` 等）。这与 ATen 模式相反，后者使用 ATen 实现的张量（`at::Tensor`）和相关的类型（`ScalarType` 等）。

- `torch::executor::Tensor`，也称为 `ETensor`，是 `at::Tensor` 的兼容子集。针对 ETensor 编写的代码可以构建在 `at::Tensor` 上。
- ETensor 本身不会拥有或分配内存。为了支持动态形状，内核可以使用客户端提供的 MemoryAllocator 分配张量数据。

(portable-kernels)=
## 便携式内核

便携式内核是与 ETensor 兼容的算子实现。由于 ETensor 与 `at::Tensor` 兼容，便携式内核可以构建在 `at::Tensor` 上，并与 ATen 内核一起用于相同的模型。便携式内核具有：

- 与 ATen 算子签名兼容
- 用便携式 C++ 编写，以便为任何目标构建
- 作为参考实现编写，优先考虑清晰度和简单性而不是优化
- 通常比 ATen 内核小得多
- 避免使用 new/malloc 动态分配内存。

## 程序

Program 描述 ML 模型的代码和数据集合。

## 程序源代码

描述程序的 Python 源代码。它可以是 Python 函数，或者是 PyTorch 的急切模式 `nn.Module` 中的方法。

## PTQ（Post Training Quantization）

一种量化技术，在模型训练完成后对其进行量化（通常为了性能优势）。与在训练期间应用量化的 QAT 不同，PTQ 在训练后应用量化流程。

## QAT（Quantization Aware Training）

量化可能会导致模型精度下降。与例如 PTQ 相比，QAT 通过在训练过程中模拟量化的影响来实现更高的精度。在训练期间，所有权重和激活值都进行“伪量化”；浮点值被舍入以模拟 int8 值，但所有计算仍然使用浮点数进行。因此，在训练期间进行的所有权重调整都会自适应到模型最终将被量化。与在训练后应用量化的PTQ不同，QAT在训练期间应用量化流程。

## 量化

一种用于在张量上执行较低精度数据（通常是 int8）的计算和内存访问的技术。量化通过降低内存使用和（通常）减少计算延迟来提高模型性能；根据硬件的不同，较低精度的计算通常会更快，例如 int8 矩阵乘法相对于 fp32 矩阵乘法。通常，量化是以牺牲模型准确性为代价的。

## 运行时

ExecuTorch 运行时（Runtime）在边缘设备上执行模型。它负责程序初始化、程序执行以及可选的销毁（释放后端拥有的资源）。

## SDK（Software Development Kit）

软件开发工具包。用户需要使用该工具来对使用 ExecuTorch 运行的程序进行性能分析、调试和可视化。

## 选择性构建

一种 API，用于仅链接到程序使用的内核来构建更精简的运行时。这可以显著减少二进制文件的大小。

(static-quantization)=
## 静态量化

静态量化（Static Quantization）是一种量化方法，其中张量是静态量化的。也就是说，在推理之前将浮点数转换为较低精度的数据类型。

## XNNPACK

一个针对 ARM、x86、WebAssembly 和 RISC-V 平台的神经网络接口算子的优化库。它是开源项目，被 PyTorch 和 ExecuTorch 使用。它是 QNNPack 库的后继者。算子同时支持浮点数和量化值。
