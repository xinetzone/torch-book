# ExecuTorch 简介

ExecuTorch 是一种端到端的方案，可为移动和边缘设备（包括可穿戴设备、嵌入式设备和微控制器）提供设备上的推理能力。它是 PyTorch Edge 生态系统的一部分，可以将 PyTorch 模型高效地部署到边缘设备上。ExecuTorch 的主要价值主张是：
- 可移植性：与各种计算平台兼容，从高端移动电话到高度受限的嵌入式系统和微控制器。
- 生产力：使开发人员能够使用相同的工具链和 SDK 从 PyTorch 模型创作和转换，调试和部署到各种平台。
秒表图标
- 性能：由于轻量级运行时并利用了 CPU、NPU 和 DSP 等全部硬件能力，为最终用户提供无缝且高性能的体验。

ExecuTorch 的主要目标之一是实现更广泛的自定义和部署能力，以支持 PyTorch 程序的广泛应用。

```{admonition} 为什么选择 ExecuTorch？
支持设备上的AI面临着独特的挑战，包括多样化的硬件、关键的功耗要求、低/无互联网连接和实时处理需求。这些限制因素历来阻碍或减缓了可扩展和高性能设备上AI解决方案的创建。我们设计了ExecuTorch，得到了Meta、Arm、苹果和高通等行业领导者的支持，旨在高度可移植并提供卓越的开发人员生产力，同时不牺牲性能。
```

```{admonition} ExecuTorch 与 PyTorch Mobile（轻量级解释器）有何不同？
PyTorch Mobile 使用 TorchScript 允许 PyTorch 模型在资源有限的设备上运行。ExecuTorch 具有明显更小的内存大小和动态内存占用，与 PyTorch Mobile 相比具有卓越的性能。此外，ExecuTorch 不依赖于 TorchScript，而是利用 PyTorch 2.0 编译器和导出功能来执行设备上的 PyTorch 模型。
```
