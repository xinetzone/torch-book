(metrics:wpsnr)=
# WPSNR

WPSNR 是 "Weighted Peak Signal-to-Noise Ratio" 的缩写，中文通常翻译为“加权峰值信噪比”。它是一种用于评估图像质量的指标，特别是在图像压缩、图像恢复（如超分辨率、去噪等）以及图像处理领域中广泛使用。WPSNR 在传统的 PSNR 基础上引入了加权因子，以更好地反映人眼对图像不同区域的敏感性。

## WPSNR 的基本概念

WPSNR 通过在计算 PSNR 时引入加权因子来评估图像质量。加权因子考虑了人眼对图像不同区域的敏感性，例如人眼对图像中心区域的细节更为敏感，而对边缘区域的细节敏感度较低。因此，WPSNR 能够更准确地反映图像在视觉上的质量。

## WPSNR 的计算公式

WPSNR 的计算公式如下：

$$
\text{WPSNR}(x, y) = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{\frac{1}{N} \sum_{i=0}^{N-1} w_i \cdot (x_i - y_i)^2} \right)
$$

其中：
- $ x $ 是原始图像。
- $ y $ 是处理后的图像。
- $ MAX_I $ 是图像像素值的最大可能值。对于8位图像，$ MAX_I = 255 $。
- $ N $ 是图像的总像素数。
- $ w_i $ 是第 $ i $ 个像素的加权因子。
- $ x_i $ 和 $ y_i $ 分别是原始图像和处理后图像在第 $ i $ 个像素处的像素值。

## 加权因子的选择

加权因子 $ w_i $ 的选择通常基于人眼对图像不同区域的敏感性。常见的加权因子选择方法包括：

1. **中心加权**：图像中心区域的加权因子较大，边缘区域的加权因子较小。
2. **视觉感知模型**：使用视觉感知模型（如对比敏感度函数，CSF）来计算每个像素的加权因子。

## WPSNR 的解释

- **高 WPSNR 值**：表示处理后的图像与原始图像在视觉上非常接近，特别是在人眼敏感的区域，图像质量高。
- **低 WPSNR 值**：表示处理后的图像与原始图像在视觉上差异较大，特别是在人眼敏感的区域，图像质量低。

## WPSNR 的优势

1. **视觉感知一致性好**：WPSNR 考虑了人眼对图像不同区域的敏感性，能够更准确地反映图像在视觉上的质量。
2. **更全面的评估**：相比于传统的 PSNR，WPSNR 能够更全面地评估图像质量，特别是在人眼敏感的区域。

## WPSNR 的局限性

虽然 WPSNR 在许多情况下能够准确反映图像质量，但它也有一些局限性：

1. **计算复杂度较高**：WPSNR 的计算涉及加权因子的计算，计算复杂度较高，尤其是在处理大图像时。
2. **加权因子选择**：加权因子的选择可能会影响最终的评估结果，不同的加权因子选择可能导致不同的 WPSNR 值。

## 总结

WPSNR 是一种基于视觉感知模型的图像质量评估指标，通过在计算 PSNR 时引入加权因子来评估图像质量。尽管 WPSNR 的计算复杂度较高，但它在许多图像处理任务中被认为是一个更准确的图像质量评估指标。