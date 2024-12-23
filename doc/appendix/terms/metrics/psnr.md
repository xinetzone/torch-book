(metrics:psnr)=
# PSNR

峰值信噪比（[Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)，简写为 PSNR）。它是一种用于评估图像质量的指标，特别是在图像压缩和图像恢复（如超分辨率、去噪等）领域中广泛使用。

## PSNR 的基本概念

PSNR 是通过比较原始图像（Ground Truth）和处理后的图像（如压缩后的图像或超分辨率重建的图像）之间的差异来评估图像质量的。具体来说，PSNR 衡量的是信号（即图像）的最大可能功率与噪声（即误差）的功率之间的比率。

### 信噪比

[信噪比](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)（英语：Signal-to-noise ratio，缩写为 SNR 或 S/N），用于比较所需信号的强度与背景噪声的强度。其定义为信号功率与噪声功率的比率，以分贝（dB）为单位表示。大于比率1:1（高于0分贝）表示信号多于噪声。信噪比通常用于描述电子信号，也可以应用在各种形式的信号，比如冰芯内的同位素量，或细胞间的生物化学信号。

$$
\text{SNR} = {P_\mathrm{signal} \over P_\mathrm{noise}} = {A_\mathrm{signal}^2 \over A_\mathrm{noise}^2 }
$$

它的单位一般使用分贝，其值为十倍对数信号与噪声功率比：

$$
\mathrm{SNR (dB)} = 10 \log_{10} \left ( {P_\mathrm{signal} \over P_\mathrm{noise}} \right ) = 20 \log_{10} \left({A_\mathrm{signal} \over A_\mathrm{noise}} \right)
$$

其中：
- $P_\mathrm{signal}$ 为信号功率（Power of Signal）。
- $P_\mathrm{noise}$ 为噪声功率（Power of Noise）。
- $A_\mathrm{signal}$ 为信号振幅（Amplitude of Signal）。
- $A_\mathrm{noise}$ 为噪声振幅（Amplitude of Noise）。

## PSNR 的计算公式

PSNR 通常以分贝（dB）为单位表示，计算公式如下：

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)
$$

其中：
- $MAX_I$ 是图像像素值的最大可能值。通常，对于8位图像，$MAX_I = 255$，对于浮点型数据 $MAX_I = 1$。更为通用的表示是，如果每个采样点用 B 位线性脉冲编码调制表示，那么 MAXI 就是 $2^B - 1$。
- $MSE$ 是均方误差（Mean Squared Error），表示原始图像和处理后图像之间的像素差异的平方的平均值。

MSE 的计算公式为：

$$
MSE = \frac{1}{m \times n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2
$$

其中：
- $I(i, j)$ 是原始图像在位置 $(i, j)$ 处的像素值。
- $K(i, j)$ 是处理后图像在位置 $(i, j)$ 处的像素值。
- $m$ 和 $n$ 分别是图像的高度和宽度。

上述计算是针对灰度图的，对于 RGB 图像，有三种计算方式：
1. 分别计算 RGB 三个通道的 PSNR，然后取平均值
2. 计算 RGB 三通道的 MSE ，然后再除以 3
3. 将图片转化为 YCbCr 格式，然后只计算 Y 分量也就是亮度分量的 PSNR

其中，第二和第三种方法比较常见。

## PSNR 的解释

- **高 PSNR 值**：表示处理后的图像与原始图像非常接近，误差很小，图像质量高。
- **低 PSNR 值**：表示处理后的图像与原始图像差异较大，误差较大，图像质量低。

PSNR值越大，表示图像的质量越好，一般来说：

1. 高于40dB：说明图像质量极好(即非常接近原始图像)
2. 30—40dB：通常表示图像质量是好的(即失真可以察觉但可以接受)
3. 20—30dB：说明图像质量差
4. 低于20dB：图像质量不可接受

## PSNR 的局限性

虽然 PSNR 是常用的图像质量评估指标，但它也有一些局限性：

1. **感知一致性差**：PSNR 主要基于像素级别的误差，而人眼对图像质量的感知不仅仅依赖于像素级别的误差，还依赖于图像的结构、纹理等特征。因此，高 PSNR 值并不一定意味着图像在视觉上看起来更好。
2. **不适用于所有场景**：PSNR 在某些情况下可能无法准确反映图像质量，例如在图像压缩中，某些压缩算法可能会在保持高 PSNR 的同时引入视觉上的伪影。

## 总结

PSNR 是一种常用的图像质量评估指标，主要用于衡量图像处理算法（如压缩、超分辨率等）的性能。它通过比较原始图像和处理后图像之间的均方误差来计算，并以分贝为单位表示。尽管 PSNR 有其局限性，但在许多应用中仍然是重要的参考指标。