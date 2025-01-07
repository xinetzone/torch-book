# TorchOpt

[TorchOpt](https://torchopt.readthedocs.io/en/latest/index.html) 是基于 PyTorch 构建的高效可微分优化库。TorchOpt 具有以下特点：

- 全面性：TorchOpt 提供了三种不同的微分模式——显式微分、隐式微分和零阶微分，以应对不同的可微分优化情境。
- 灵活性：TorchOpt 为用户提供了函数式和面向对象两种 API 风格，以满足用户的不同偏好。用户可以选择类似 JAX 或 PyTorch 的风格来实现可微分优化。
- 高效性：TorchOpt 提供了（1）CPU/GPU加速的可微分优化器；（2）基于RPC的分布式训练框架；（3）快速树操作，这些功能极大地提高了双层优化问题的训练效率。