# depyf

在了解 [`depyf`](https://depyf.readthedocs.io/en/latest/) 的使用方法之前，推荐您先阅读 [`torch.compile` 示例教程](walk-through)，以便理解 `depyf` 如何帮助您。

`depyf` 旨在解决 {func}`torch.compile` 的两个痛点：

1. {func}`torch.compile` 转换 Python 字节码，但很少有开发者能读懂 Python 字节码（除非你的大脑里有一台堆栈机……），从而理解发生了什么。`depyf` 帮助将转换后的字节码反编译回 Python 源代码，使开发者能够理解 {func}`torch.compile` 是如何转换他们的代码的。这极大地帮助用户调整他们的代码以适应 {func}`torch.compile`，使他们能够编写对 {func}`torch.compile` 友好的代码。
2. {func}`torch.compile` 动态生成许多函数，这些函数只能作为黑盒子运行。用户无法逐行调试代码。`depyf` 帮助将源代码导出到文件中，并将这些函数与源代码文件链接起来，这样用户就可以使用调试器逐行调试这些函数了。这极大地帮助用户理解 {func}`torch.compile` 并在训练过程中调试如 `NaN` 等问题。

采用从教程示例中的工作流程：![](https://depyf.readthedocs.io/en/latest/_images/dynamo-workflow-with-depyf.svg)

`depyf` 有助于：

- 提供上述工作流程的源代码描述，以便用户能够轻松理解。（实际的工作流程发生在 C 语言中，并在 CPython 解释器内进行，提供 Python 源代码描述的工作流程，以便用户可以更容易地理解。）
- 生成转换后的字节码和恢复函数的源代码。
- 将计算图计算函数与磁盘上的代码链接起来，以便调试器可以逐步执行代码。

`depyf` 的主要用途涉及两个上下文管理器，建议在调试器中启动脚本：

```python
import torch

@torch.compile
def function(inputs):
    x = inputs["x"]
    y = inputs["y"]
    x = x.cos().cos()
    if x.mean() > 0.5:
        x = x / 1.1
    return x * y

shape_10_inputs = {"x": torch.randn(10, requires_grad=True), "y": torch.randn(10, requires_grad=True)}
shape_8_inputs = {"x": torch.randn(8, requires_grad=True), "y": torch.randn(8, requires_grad=True)}

import depyf
with depyf.prepare_debug("./debug_dir"):
    # warmup
    for i in range(100):
        output = function(shape_10_inputs)
        output = function(shape_8_inputs)
# the program will pause here for you to set breakpoints
# then you can hit breakpoints when running the function
with depyf.debug():
    output = function(shape_10_inputs)
```

第一个上下文管理器 {func}`depyf.prepare_debug` 接受一个目录路径作为参数，用于将所有源代码转储至该目录。在这个上下文管理器中，PyTorch 的所有内部细节将被 `depyf` 挂钩，它会自动为你转储必要的源代码。

第二个上下文管理器 {func}`depyf.debug` 无需任何参数，它仅禁用新的编译条目。一旦进入此上下文管理器，程序将会暂停，你可以浏览指定目录下（本例中为 `"./debug_dir"`）的所有源代码。入口文件是 `full_code_for_xxx.py`。你可以在这些文件中设置断点。最重要的是，你在这个上下文管理器下设置的断点可以被命中。你可以逐行调试代码，以排查可能的 `NaN` 值或理解你的代码发生了什么。

下图展示了 `depyf` 的两个典型用法，并列出了所有生成的文件。![](https://depyf.readthedocs.io/en/latest/_images/usage.svg)

```{toctree}
:hidden:

walk-through
```