# {mod}`ultralytics` 框架

YOLO 系列模型

模型|锚框|输入|主干（Backbone）|颈部（Neck）|预测/训练
:-|:-|:-|:-|:-|:-
YOLOv1|锚框（$7 \times 7$ grids，2 anchors）|resize($(448, 448, 3)$)： 训练是 $224 \times 224$，测试是 $448 \times 448$；|GoogLeNet（24*Conv+2*FC+reshape；Dropout防止过拟合；最后一层使用线性激活函数，其余层都使用ReLU激活函数）；|无|IOU_Loss、nms；一个网格只预测了2个框，并且都属于同一类；全连接层直接预测bbox的坐标值；
YOLOv2|锚框（$13 \times 13$ grids，5 anchors 通过k-means选择先验框）|resize（(416, 416, 3)）：416/32=13，最后得到的是奇数值有实际的中心点；在原训练的基础上又加上了（10个epoch）的448x448高分辨率样本进行微调；|Darknet-19（19*Conv+5*MaxPool+AvgPool+Softmax；没有FC层，每一个卷积后都使用BN和ReLU防止过拟合（舍弃dropout）；提出passthrough层：把高分辨率特征拆分叠加大到低分辨率特征中，进行特征融合，有利于小目标的检测）；|无|IOU_Loss、nms；一个网络预测5个框，每个框都可以属于不同类；预测相对于anchor box的偏移量；多尺度训练（训练模型经过一定迭代后，输入图像尺寸变换）、联合训练机制；
YOLOv3|锚框（$13 \times 13$ grids，9 anchors：三种尺度*三种宽高比）|resize (608, 608, 3)|Darknet-53（53*Conv，每一个卷积层后都使用BN和Leaky ReLU防止过拟合，残差连接）；|FPN（多尺度检测，特征融合）|IOU_Loss、nms；多标签预测（softmax分类函数更改为logistic分类器）；
YOLOv4|锚框|resize(608, 608, 3)、Mosaic数据增强、SAT自对抗训练数据增强|CSPDarknet53（CSP模块：更丰富的梯度组合，同时减少计算量、跨小批量标准化（CmBN）和Mish激活、DropBlock正则化（随机删除一大块神经元）、采用改进SAM注意力机制：在空间位置上添加权重）；|SPP（通过最大池化将不同尺寸的输入图像变得尺寸一致）、PANnet（修改PAN，add替换成concat）|CIOU_Loss、DIOU_nms；自对抗训练SAT：在原始图像的基础上，添加噪音并设置权重阈值让神经网络对自身进行对抗性攻击训练；类标签平滑：将绝对化标签进行平滑（如：$[0,1] \to [0.05,0.95]$），即分类结果具有一定的模糊化，使得网络的抗过拟合能力增强；
YOLOv5|锚框|resize(608, 608, 3)、Mosaic数据增强、自适应锚框计算、自适应图片缩放|CSPDarknet53（CSP模块，每一个卷积层后都使用BN和Leaky ReLU防止过拟合，Focus模块）；|SPP、PAN|GIOU_Loss、DIOU_Nms；跨网格匹配（当前网格的上、下、左、右的四个网格中找到离目标中心点最近的两个网格，再加上当前网格共三个网格进行匹配）
YOLOX|无锚框|resize(608, 608, 3)|Darknet-53|SPP、FPN|CIOU_Loss、DIOU_Nms、Decoupled Head、SimOTA标签分配策略；
YOLOv6|无锚框|resize(640, 640, 3)|EfficientRep Backbone（Rep算子）|SPP、Rep-PAN Neck|SIOU_Loss、DIOU_Nms、Efficient Decoupled Head、SimOTA标签分配策略；
YOLOv7|锚框|resize(640, 640, 3)|Darknet-53（CSP模块替换了ELAN模块；下采样变成MP2层；每一个卷积层后都使用BN和SiLU防止过拟合）；|SPP、PAN|CIOU_Loss、DIOU_Nms、SimOTA标签分配策略、带辅助头的训练（通过增加训练成本，提升精度，同时不影响推理的时间）；
YOLOv8|无锚框|resize(640, 640, 3)|Darknet-53（C3模块换成了C2F模块）|SPP、PAN|CIOU_Loss、DFL_Loss、DIOU_Nms、TAL标签分配策略、Decoupled Head；


- YOLOv9 引入了 PGI，这是一种使用辅助可逆分支生成可靠梯度的新方法。此辅助分支为计算目标函数提供了完整的输入信息，从而确保用于更新网络权重的梯度更具信息量。辅助分支的可逆性质确保在前馈过程中不会丢失任何信息。YOLOv9 还提出了 GELAN，这是一种新的轻量级架构，旨在最大化信息流并促进获取相关信息进行预测。GELAN 是 ELAN 架构的通用版本，可以利用任何计算块，同时尽可能保持效率和性能。研究人员基于梯度路径规划设计了它，确保信息在网络中高效流动。YOLOv9 通过关注信息流和梯度质量，为目标检测提供了全新的视角。PGI 和 GELAN 的引入使 YOLOv9 有别于其前代产品。这种对深度神经网络中信息处理基础的关注，可以提高性能，并更好地解释目标检测中的学习过程。
- YOLOv10 通过 NMS-Free 检测消除了非最大抑制 (NMS) 后处理的需要。
    这不仅提高了推理速度，还简化了部署过程。YOLOv10 引入了一些关键功能，例如 NMS-Free 训练和整体设计方法，使其在所有指标上都表现出色。NMS–Free 检测：YOLOv10 基于一致的双重分配提供了一种新颖的 NMS-Free 训练策略。它采用双重标签分配（一对多和一对一）和一致的匹配度量，在训练期间提供丰富的监督，同时在推理期间消除 NMS。在推理过程中，仅使用一对一的 head，从而实现 NMS-Free 检测。
    整体效率-准确度驱动设计：YOLOv10 采用整体方法进行模型设计，优化各个组件以提高效率和准确度。它引入了轻量级分类 head、空间通道解耦下采样和等级引导块设计，以降低计算成本。
- YOLO11 引入了 C3k2 块和 C2PSA 块等新组件，有助于改进特征提取和处理。这导致性能略有提高，但模型的参数却少得多。以下是 YOLO11 的主要功能。
    C3k2 块：YOLO11 引入了 C3k2 块，这是跨阶段部分 (CSP) 瓶颈的计算高效实现。它取代了 backbone 和 neck 中的 C2f 块，并使用两个较小的卷积代替一个大卷积，从而缩短了处理时间。
    C2PSA 块：在空间金字塔池化 - 快速 (SPPF) 块之后引入跨阶段部分空间注意 (C2PSA) 块，以增强空间注意。这种注意机制使模型能够更有效地关注图像中的重要区域，从而有可能提高检测准确性。

```{toctree}
yolov1
yolov2
yolov3
ref/index
```
