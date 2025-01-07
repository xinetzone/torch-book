# 常见问题({func}`torch.compile`)

如果缺失 `filelock`，则：

```bash
conda install filelock
```

## NumPy 是否与 {func}`torch.compile` 一起使用？

从 2.1 版本开始，{func}`torch.compile` 能够理解在 NumPy 数组上工作的原生 NumPy 程序，以及通过 `x.numpy()`、`torch.from_numpy` 和相关函数在 PyTorch 和 NumPy 之间转换的混合 PyTorch-NumPy 程序。
