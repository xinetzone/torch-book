# Colab 训练

## 安装 Nightly 构建

因为您将使用 PyTorch 的实验部件，所以建议安装最新版本的 ``torch`` 和 ``torchvision``。您可以找到关于[本地安装的最新说明](https://pytorch.org/get-started/locally/)。例如，在没有 GPU 支持的情况下安装：

```shell
pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
# 对于 CUDA 支持使用 https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```

或者：

```shell
!yes y | pip uninstall torch torchvision
!yes y | pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```
