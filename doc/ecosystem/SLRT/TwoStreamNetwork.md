# 双流 SLT

一种用于手语识别和翻译的双流网络，包括[官方实现](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)：
* [A Simple Multi-modality Transfer Learning Baseline for Sign Language Translation, CVPR2022](https://arxiv.org/abs/2203.04287)
* [Two-Stream Network for Sign Language Recognition and Translation, NeurIPS2022](https://arxiv.org/abs/2211.01367).

手语翻译（Sign Language Translation，简称 SLT）和手语识别（Sign Language Recognition，简称 SLR）面临着数据稀缺的问题。为了缓解这个问题，首先提出了[一种简单多模态迁移学习基线（baseline）用于 SLT](https://arxiv.org/abs/2203.04287)，该基线通过从通用领域到领域内逐步预训练模块，利用大规模通用领域数据集的额外监督，并最终进行多模态联合训练。这种简单而有效的基线实现了强大的翻译性能，显著优于以往的工作。

进一步提出了[一种用于 SLR 和 SLT 的双流网络](https://arxiv.org/abs/2211.01367)，该网络将人体关键点领域的知识融入视觉编码器。双流网络在 SLR 和 SLT 基准测试中取得了 SOTA 性能（ `18.8 WER on Phoenix-2014 and 19.3 WER on Phoenix-2014T, 29.0 BLEU4 on Phoenix-2014T, and 25.8 BLEU4 on CSL-Daily`）。