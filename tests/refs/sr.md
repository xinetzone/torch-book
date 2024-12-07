# 超分辨率概述(待更)

参考：[图像超分辨率重建综述](https://pdf.hanspub.org/CSA20240200000_67370344.pdf)

{term}`超分辨率`(SR) 目标：将低分辨率图像提升至高分辨率，以改善图像质量和细节。

从输入图像数量的角度，可以将图像超分辨率重建方法分为 {term}`SISR`(单图超分) 和 {term}`MISR`(多图超分) {footcite:p}`9044873`。MISR 方法利用多个 LR 图像来提高单一图像的分辨率，以增强整体图像的质量，逐渐发展为 {term}`视频超分辨率` {footcite:p}`9870558`。

## 单图像超分辨率重建方法

SISR 可以分为 {term}`盲超分辨率` (BSR) 和 {term}`非盲超分辨率` {footcite:p}`8723565` (NBSR)。

## 非盲超分辨率重建方法

基于深度学习的 SISR 将给定的单张低分辨率图像利用超分辨率方法转换为高分辨率图像，而无需额外的多个图像作为参考。2014 年 {cite:author}`dong2015imagesuperresolutionusingdeep` 首次将深度学习方法引入图像超分重建任务，提出了超分辨率卷积神经网络（Super-Resolution Convolutional Neural Network, SRCNN）{footcite:p}`dong2015imagesuperresolutionusingdeep`，借助卷积神经网络(Convolutional Neural Network, CNN)强大的学习能力，取得了优于传统方法的效果，奠定了基于卷积神经网络的非盲图像超分辨率的理论基础。