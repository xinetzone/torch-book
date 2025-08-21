# 扩展 torch Python API

参考：[extending-torch-python-api](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api)

你可以通过定义自定义类并为其定义匹配 `Tensor` 的方法来创建自定义类型以模拟 `Tensor` 。但如果希望能够在顶级 {func}`torch.add` 命名空间中的函数（如 `torch` ）接受 `Tensor` 操作数时传递这些类型呢？

如果你的自定义 Python 类型定义了名为 `__torch_function__` 的方法，当自定义类的实例传递给 `torch` 命名空间中的函数时，PyTorch 将调用你的 `__torch_function__` 实现。这使得可以为 `torch` 命名空间中的任何函数定义自定义实现，而你的 `__torch_function__` 实现可以调用这些函数，从而使用户能够使用他们已经为 `Tensor` 编写的现有 PyTorch 工作流来利用你的自定义类型。这不仅适用于与 Tensor 无关的“鸭子类型”，还适用于用户定义的 `Tensor` 的子类。

```{toctree}
TorchFunction
subclassing-tensor
torch-dispatch
```
