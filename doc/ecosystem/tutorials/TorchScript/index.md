# TorchScript

参考：[jit](https://pytorch.org/docs/stable/jit.html)

TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法。任何 TorchScript 程序都可以从 Python 进程中保存，并在没有 Python 依赖的进程中加载。

可以逐步将模型从纯 Python 程序过渡到可以独立于 Python 运行的 TorchScript 程序，例如在独立的 C++ 程序中运行。这使得可以在 PyTorch 中使用熟悉的 Python 工具训练模型，然后通过 TorchScript 将模型导出到生产环境中，因为在这些环境中，Python 程序可能由于性能和多线程的原因而处于不利地位。

对于 TorchScript 的温和介绍，请参阅 [TorchScript 简介教程](intro)。

对于将 PyTorch 模型转换为 TorchScript 并在 C++ 中运行的端到端示例，请参阅 [在 C++ 中加载 PyTorch 模型教程](https://pytorch.org/tutorials/advanced/cpp_export.html)。

```{toctree}
:hidden:

intro
jit
```
