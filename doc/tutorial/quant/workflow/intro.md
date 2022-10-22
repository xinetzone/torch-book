# 量化流程概述

参考：[量化基础](https://pytorch.org/docs/master/quantization.html) & [introduction-to-quantization-on-pytorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

在较低的层次上，PyTorch 提供了一种表示量化张量并对它们执行运算的方法。它们可以用来直接构建模型，以较低的精度执行全部或部分计算。提供了更高层次的 API，结合了 FP32 模型转换的典型工作流程，以最小的精度损失降低精度。

量化需要用户了解三个概念：

- `QConfig`：指定量化 Weight 和 Activation 的配置方式
- 后端：用于支持量化的内核
- 引擎（{mod}`~torch.backends.quantized.engine`）：指定执行时所需后端

```{note}
在准备量化模型时，必须确保 `qconfig` 和用于量化计算的引擎与将在其上执行模型的后端匹配。
```

量化的后端：

- AVX2 X86 CPU：[`'fbgemm'`](https://github.com/pytorch/FBGEMM)
- ARM CPU（常用于手机和嵌入式设备）：[`'qnnpack'`](https://github.com/pytorch/QNNPACK)

相应的实现会根据 PyTorch 构建模式自动选择，不过用户可以通过将 `torch.backends.quantization.engine` 设置为 `'fbgemm'` 或 `'qnnpack'` 来覆盖这个选项。

量化感知训练（通过 {mod}`~torch.quantization.FakeQuantize`，它模拟 FP32 中的量化数字）支持 CPU 和 CUDA。

`qconfig` 控制量化传递期间使用的观测器器类型。当对线性和卷积函数和模块进行打包权重时，`qengine` 控制是使用 `fbgemm` 还是 `qnnpack` 特定的打包函数。例如：

```python
# set the qconfig for PTQ
qconfig = torch.quantization.get_default_qconfig('fbgemm')
# or, set the qconfig for QAT
qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'fbgemm'
```

## API 概述

PyTorch 提供了两种不同的量化模式：Eager 模式量化和 FX 图模式量化。

Eager 模式量化是 beta 特性。用户需要进行融合，并手动指定量化和反量化发生的位置，而且它只支持模块而不支持函数。

FX 图模式量化是 PyTorch 中新的自动量化框架，目前它是原型特性。它通过添加对函数的支持和量化过程的自动化，对 Eager 模式量化进行了改进，尽管人们可能需要重构模型，以使模型与 FX Graph 模式量化兼容（通过 `torch.fx` 符号可追溯）。注意 FX 图模式量化预计不会在任意工作模型由于模型可能不是符号可追溯，我们会将其集成到域库 torchvision 和用户将能够量化模型类似于支持域的库与 FX 图模式量化。对于任意的模型，我们将提供一般的指导方针，但要让它实际工作，用户可能需要熟悉 `torch.fx`，特别是如何使模型具有符号可追溯性。

新用户的量化鼓励首先尝试 FX 图模式量化，如果它不工作，用户可以尝试遵循[使用 FX 图模式量化的指导方针](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)或回落到 Eager 模式量化。

支持三种类型的量化：

1. 动态量化（通过读取/存储在浮点数中的激活量化权重，并量化用于计算。）
2. 静态量化（权重量化，激活量化，校准所需的后训练）
3. 静态量化感知训练（权重量化、激活量化、训练时建模的量化数值）

PyTorch 设计了 {mod}`torch.ao.quantization` 来适应 PyTorch 框架。这意味着：

- PyTorch 具有与 [量化张量](QuantizedTensor) 对应的数据类型，它们具有张量的许多特性。
- 可以用量化张量编写内核，就像浮点张量的内核一样，来定制它们的实现。PyTorch 支持 `quantized` 模块，用于通用运算，作为 `torch.nn.quantized` 和 `torch.nn.quantized.dynamic` 名称空间的一部分。
- 量化与 PyTorch 的其余部分兼容：量化模型是 traceable 和 scriptable。对于服务器和移动后端，量化方法实际上是相同的。可以很容易地在模型中混合量化和浮点运算。
- 浮点张量到量化张量的映射可以通过用户定义的 observer/fake-quantization 块进行定制。PyTorch 提供了默认的实现，应该适用于大多数用例。

在 PyTorch 中开发了三种量化神经网络的技术，作为 {mod}`torch.ao.quantization` 中量化工具的一部分。

## 动态量化简介

PyTorch 支持的最简单的量化方法称为动态量化（dynamic quantization）。这不仅涉及到将权值转换为 int8（就像所有量化变量中发生的那样），还涉及到在执行计算之前动态地将激活转换为 int8（因此是动态的）。因此，计算将使用高效的 int8 矩阵乘法和卷积实现来执行，从而获得更快的计算速度。但是，激活是用浮点格式读写到内存中的。

在 PyTorch 中有简单的动态量化API `torch.quantization.quantize_dynamic`，它接受模型以及几个其他参数，并生成量化模型。

[端到端教程](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)为 BERT 模型演示了这一点；虽然教程很长，包含了加载预训练模型和其他与量化无关的概念的部分，但量化 BERT 模型的部分很简单。

```python
from torch.quantization.quantize import quantize_dynamic

model_dynamic_quantized = quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
```

其中 `qconfig_spec` 指定要应用量化的 `model` 中的子模块名称列表。

```{warning}
动态量化的一个重要限制是，虽然它是最简单的工作流程，如果你没有预先训练的量化模型准备使用，它目前在 `qconfig_spec` 中只支持 `nn.Linear` 和 `nn.LSTM`，这意味着你将不得不使用静态量化或量化感知训练，稍后讨论，以量化其他模块，如 `nn.Conv2d`。
```

```{note}
训练后动态量化示例：

- [Bert 例子](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
- [LSTM 模型例子](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html#test-dynamic-quantization)
- [LSTM demo 例子](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#do-the-quantization)。
```

## 后训练静态量化简介

通过将网络转换为同时使用整数算法和 int8 内存访问，可以进一步提高性能（延迟）。静态量化执行额外的步骤，首先通过网络输入批数据，并计算不同激活的结果分布（具体来说，这是通过在记录这些分布的不同点插入 `observer` 模块来完成的）。这一信息用于确定在推理时应该如何具体地量化不同的激活（一种简单的技术是将整个激活范围简单地划分为 256 个级别）。

重要的是，这个额外的步骤允许在运算之间传递量化的值，而不是在每个运算之间将这些值转换为浮点数（然后再转换为整数），从而显著提高了速度。

允许用户优化静态量化：

- `Observer`：可以定制观测者模块，该模块指定在量化之前如何收集统计数据，以尝试更高级的方法来量化数据。
- 算子融合：可以将多个算子融合为单个算子，节省内存访问，同时提高运算的数值精度。
- 逐通道量化：可以在卷积/线性层中独立量化每个输出通道的权值，这可以在几乎相同的速度下获得更高的精度。

```{note}
1. 使用 {func}`torch.ao.quantization.fuse_modules` 融合模块：
2. 使用 {func}`torch.quantization.prepare` 插入观测者
3. 最后，量化本身是使用 {func}`torch.ao.quantization.convert` 完成的
```

```python
from torch.quantization.quantize import prepare, convert
from torch.quantization.qconfig import get_default_qconfig
from torchvision.models.quantization import resnet18 as qresnet18

# 设置量化配置
backend = "fbgemm" # 若为 x86，否则为 'qnnpack' 
model.qconfig = get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# 插入观测器
model_static_quantized = prepare(model, inplace=False)
# 转换为量化版本
model_static_quantized = convert(model_static_quantized, inplace=False)
```

[这里](https://pytorch.org/docs/stable/quantization.html#quantization-api-summary)有一个完整的模型定义和静态量化的例子。[这里](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)有一个专门的静态量化教程。

## 量化感知训练简介

量化感知训练（Quantization-aware training，简称 QAT）是第三种方法，通常是这三种方法中准确度最高的一种。使用 QAT，所有的权值和激活都在前向和后向训练过程中被伪量化：也就是说，浮点值被舍入以模拟 int8 值，但所有的计算仍然使用浮点数完成。因此，在意识到模型最终将被量化的情况下，对训练过程中的所有权值进行调整；因此，在量化后，该方法通常比其他两种方法获得更高的精度。

1. `torch.ao.quantization.prepare_qat` 插入伪量化模块来建模量化。
2. 模仿静态量化 API，一旦训练完成，`torch.quantization.convert` 为真正的量化模型。

例如，在[端到端示例](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)中，我们将预训练的模型作为 `qat_model` 加载进去，然后使用 `qat_model` 执行量化感知训练：

```python
# specify quantization config for QAT
qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')

# prepare QAT
torch.ao.quantization.prepare_qat(qat_model, inplace=True)

# convert to quantized version, removing dropout, to check for accuracy on each
epochquantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
```

要启用量化感知训练的模型，请在模型定义的 `__init__` 方法中定义 `QuantStub` 和 `DeQuantStub`，以将张量从浮点类型转换为量化类型，反之亦然。

```python
from torch import nn
from torch.quantization.stubs import QuantStub, DeQuantStub

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # 其他运算
        x = self.dequant(x)
        return x
```

然后在模型定义的前向方法的开头和结尾分别调用 `x = self.quant(x)` 和 `x = self.dequant(x)`。

要进行量化感知训练，请使用下面的代码片段:

```python
from torch.quantization.quantize import prepare_qat, convert
from torch.quantization.qconfig import get_default_qat_qconfig

model = MyModel()
model.qconfig = get_default_qat_qconfig(backend)
model_qat = prepare_qat(model, inplace=False)
# QAT
model_qat = convert(model_qat.eval(), inplace=False)
```

有关量化感知训练的更多详细示例，请参阅[此处](https://pytorch.org/docs/master/quantization.html#quantization-aware-training)和[此处](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training)。

在使用上面的步骤之一生成量化模型之后，在模型可以在移动设备上运行之前，它需要进一步转换为 TorchScript 格式，然后针对移动端应用程序进行优化。请参阅[脚本和优化移动端食谱](https://pytorch.org/tutorials/recipes/script_optimized.html)的详细信息。

## 设备和算子支持

量化支持仅限于可用算子的子集，具体取决于所使用的方法，有关支持算子的列表，请参阅 [文档](https://pytorch.org/docs/stable/quantization.html)。

可用算子和量化数值的集合也取决于用于运行量化模型的后端。目前，量化算子仅支持以下后端 CPU 推理：x86 和 ARM。两个量化配置（如何量化张量和量化内核（量化张量的算法）都是后端依赖的）。可以通过执行来指定后端：

```python
import torch
backend='fbgemm'
# 'fbgemm' for server, 'qnnpack' for mobile
my_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
# prepare and convert model
# Set the backend on which the quantized kernels need to be run
torch.backends.quantized.engine=backend
```

然而，量化感知训练发生在全浮点，可以在 GPU 或 CPU 上运行。量化感知训练通常只在训练后的静态或动态量化不能产生足够的准确性时使用 CNN 模型。这可能发生在高度优化以实现小尺寸的模型（如 Mobilenet）上。

## 小结

目前，算子的覆盖范围是有限的，可能会限制下表中列出的选择：下表提供了一个指导原则。

模型类型|首选方案|缘由
:-|:-|:-
LSTM/RNN|动态量化|吞吐量受权重的计算或者内存带宽支配
BERT/Transformer|动态量化|吞吐量受权重的计算或者内存带宽支配
CNN	|静态量化|吞吐量受激活的内存带宽限制
CNN	|量化感知训练|在静态量化无法达到精度的情况下

```{note}
- Dynamic Quantization：动态量化
- Static Quantization：静态量化
- Quantization Aware Training：量化感知训练
- activations：激活
- throughput：吞吐量
- bandwidth：带宽
```

```{tip}
如果您正在处理序列数据，则从 [LSTM](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html) 或 [BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) 的动态量化开始。如果您正在处理图像数据，建议从[带量化的迁移学习教程](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)开始。然后你可以探索[静态后训练量化](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)。如果你发现训练后量化的准确率下降太高，那么尝试[量化感知训练](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)。
```
