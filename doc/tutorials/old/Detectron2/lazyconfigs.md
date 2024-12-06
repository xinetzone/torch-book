# 延迟配置

```{admonition} 概要
通过使用递归实例化来创建对象，避免了将大量的配置传递到许多地方，因为 `cfg` 只传递给 `instantiate`。这有以下好处:

1. 非侵入式(non-intrusive)的: 要构造的对象是与配置无关的常规 Python 函数/类。他们甚至可以用于其他库。利用 `{"_target_": "torch.nn.Conv2d", "in_channels": 10, "out_channels": 10, "kernel_size": 1}` 定义了卷积层
2. 更加明确调用什么函数/类，以及它们使用什么参数。
3. `cfg` 不需要预定义的键和结构。只要它转换为有效的代码，它就是有效的。这就提供了更多的灵活性。
4. 你仍然可以像以前一样传递大的字典作为参数。
```

传统的基于 yacs 的配置系统提供基本的标准功能。然而，它并没有为许多新项目提供足够的灵活性。detectron2 开发了一种替代的非侵入式配置系统，可用于 detectron2 或任何其他复杂的项目。

## Python 语法

配置对象仍然是字典。不使用 YAML 定义字典，而是直接在 Python 中创建字典。这为用户提供了 YAML 中不存在的以下功能:

- 使用 Python 轻松操作字典(添加和删除)。
- 编写简单的算术或调用简单的函数。
- 使用更多的数据类型/对象。
- 使用熟悉的 Python 导入语法导入/合成其他配置文件。

Python 配置文件可以这样加载:

```python
# config.py:
a = dict(x=1, y=2, z=dict(xx=1))
b = dict(x=3, y=4)

# my_code.py:
from detectron2.config import LazyConfig
cfg = LazyConfig.load("path/to/config.py")  # an omegaconf dictionary
assert cfg.a.z.xx == 1
```

{meth}`~detectron2.config.LazyConfig.load` 后, `cfg` 是字典，它包含配置文件全局作用域中定义的所有字典。注意:

- 在加载过程中，所有字典都转换为 [omegaconf](https://omegaconf.readthedocs.io) 配置对象。这允许访问 omegaconf 特性，例如它的[访问语法](https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#access-and-manipulation)和[插值](https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation)。
- 在 `config.py` 中绝对导入的工作原理与在常规 Python 中相同。
- 相对导入只能从配置文件中导入字典。它们只是 {meth}`~detectron2.config.LazyConfig.load_rel` 的语法糖。它们可以在相对路径上加载 Python 文件，而不需要 `__init__.py`。

{meth}`~detectron2.config.LazyConfig.save` 可以将配置对象保存到 YAML。注意，如果配置文件中出现了不可序列化的对象(例如 lambdas)，这并不总是成功的。是否牺牲 save 的能力来换取灵活性取决于用户。

## 递归模板具现化

LazyConfig 系统大量使用递归实例化，这种模式使用字典来描述对函数/类的调用。该词典包括:

1. 包含可调用对象路径的 `_target_` 键，如 `module.submodule.class_name`。
2. 表示传递给可调用对象的参数的其他键。参数本身可以使用递归实例化来定义。

辅助函数 {class}`~detectron2.config.LazyCall` 帮助创建这样的字典。

```python
from detectron2.config import LazyCall as L
from my_app import Trainer, Optimizer
cfg = L(Trainer)(
  optimizer=L(Optimizer)(
    lr=0.01,
    algo="SGD"
  )
)
```

创建像这样的字典:

```python
cfg = {
  "_target_": "my_app.Trainer",
  "optimizer": {
    "_target_": "my_app.Optimizer",
    "lr": 0.01, "algo": "SGD"
  }
}
```

通过使用这样的字典表示对象，一般实例化函数可以将它们转换为实际对象，即:

```python
from detectron2.config import instantiate
trainer = instantiate(cfg)
# equivalent to:
# from my_app import Trainer, Optimizer
# trainer = Trainer(optimizer=Optimizer(lr=0.01, algo="SGD"))
```

这个模式非常强大，足以描述非常复杂的对象，例如, 递归实例化中描述的全掩码 R-CNN

```python
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from ..data.constants import constants

model = L(GeneralizedRCNN)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=constants.imagenet_bgr256_std,
    input_format="BGR",
)
```

还有一些对象或逻辑不能简单地用字典来描述，例如重用的对象或方法调用。它们可能需要一些重构来处理递归实例化。

## 使用模型动物园 LazyConfigs

使用 LazyConfig 系统在模型动物园中提供了一些配置，例如:

* [common baselines](https://github.com/facebookresearch/detectron2/blob/main/configs/common/).

* [new Mask R-CNN baselines](https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/)

在安装 detectron2 之后，可以通过模型动物园 API {func}`~detectron2.model_zoo.get_config` 加载。

使用这些作为参考，您可以自由地为自己的项目定义自定义配置结构/字段，只要您的训练脚本能够理解它们。尽管如此，为了保持一致性，模型动物园配置仍然遵循一些简单的约定，例如 `cfg.model` 定义了模型对象 `cfg.dataloader.{train,test}` 定义数据加载器对象，`cfg.train` 包含键值形式的训练选项。

除了 {func}`print`，查看配置结构的更好方法是这样的:

```python
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
print(LazyConfig.to_py(get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py")))
```

从输出中更容易找到相关的更改选项，例如 `dataloader.train.total_batch_size` 为批大小或 `optimizer.lr` 为基础学习率。

参考脚本 [`tools/lazyconfig_train_net.py`](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)，它可以训练/评估模型动物园配置。它还展示了如何支持命令行值重写。

为了演示新系统的强大功能和灵活性，展示了[简单的配置文件](https://github.com/facebookresearch/detectron2/blob/main/configs/Misc/torchvision_imagenet_R_50.py)可以让 detectron2 从 torchvision 训练 ImageNet 分类模型，即使 detectron2 不包含关于 ImageNet 分类的特性。这可以作为在其他深度学习任务中使用 detectron2 的参考。
