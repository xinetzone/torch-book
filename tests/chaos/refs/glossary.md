# 术语表

```{glossary}
PTQ
训练后量化
    简称 PTQ（Post Training Quantization）：权重量化，激活量化，需要借助数据在训练后进行校准。
    ```
    # 原始模型
    ## 全部的张量和计算均在浮点上进行
    previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                        /
        linear_weight_fp32

    # 静态量化模型
    ## weights 和 activations 在 int8 上
    previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                        /
    linear_weight_int8
    ```

QAT
静态量化感知训练
    简称 QAT（static quantization aware training）：权重量化，激活量化，在训练过程中的量化数值进行建模。
    ```
    # 原始模型
    # 全部张量和计算均在浮点上
    previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                        /
        linear_weight_fp32

    # 在训练过程中使用 fake_quants 建模量化数值
    previous_layer_fp32 -- fq -- linear_fp32 -- activation_fp32 -- fq -- next_layer_fp32
                            /
    linear_weight_fp32 -- fq

    # 量化模型
    # weights 和 activations 在 int8 上
    previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                        /
    linear_weight_int8
    ```

浮点模型
    模型的 权重 和 激活 均为浮点类型（如 {data}`torch.float32`, {data}`torch.float64`）。

量化模型
    模型的 权重 和 激活 均为量化类型（如 {data}`torch.qint32`, {data}`torch.qint8`, {data}`torch.quint8`, {data}`torch.quint2x4`, {data}`torch.quint4x2`）。
```