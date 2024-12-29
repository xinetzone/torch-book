# 扩展 Detectron2 的默认值

研究就是用新的方法做事。这给如何在代码中创建抽象带来了压力，这对任何大规模的研究工程项目都是挑战:

1. 一方面，它需要非常薄的抽象，以允许以新的方式做任何事情的可能性。打破现有的抽象并用新的抽象替换它们应该是相当容易的。
2. 另一方面，这样的项目也需要合理的高级抽象，这样用户就可以很容易地以标准的方式做事，而不必过于担心只有某些研究人员才关心的细节。

在 detectron2 中，有两种类型的接口共同解决这种紧张关系:

1. 接受从 YAML 文件创建的 config (`cfg`) 参数的函数和类(有时带有少量额外参数)。

    这样的函数和类实现了“标准默认”行为: 它将从给定的配置中读取所需的内容，并执行“标准”操作。用户只需要加载特殊制作的配置并传递它，而不必担心使用了哪些参数以及它们都意味着什么。

    有关详细教程，请参阅 [Yacs 配置](configs)。

2. 具有定义良好的显式参数的函数和类。

    每一个都是整个系统的一个小构件。它们需要用户的专业知识来理解每个参数应该是什么，并且需要更多的努力来拼凑成更大的系统。但它们可以用更灵活的方式拼接在一起。

    当需要实现 detectron2 中包含的“标准默认值”不支持的内容时，可以重用这些定义良好的组件。

    [LazyConfig 系统](lazyconfigs) 依赖于这样的函数和类。

3. 一些函数和类是用 {func}`detectron2.config.configurable` 实现的——它们可以用配置调用，也可以用显式参数调用，或者两者混合调用。它们的显式参数接口目前还处于试验阶段。

    举个例子，Mask R-CNN 模型可以通过以下方式构建:

    1. Config-only:

    ```python
    # load proper yaml config file, then
    model = build_model(cfg)
    ``` 
    2. 混合配置和附加参数覆盖:
    ```python
    model = GeneralizedRCNN(
    cfg,
    roi_heads=StandardROIHeads(cfg, batch_size_per_image=666),
    pixel_std=[57.0, 57.0, 57.0])
    ```
    3. 完整的显式参数:
    ```python
    model = GeneralizedRCNN(
        backbone=FPN(
            ResNet(
                BasicStem(3, 64, norm="FrozenBN"),
                ResNet.make_default_stages(50, stride_in_1x1=True, norm="FrozenBN"),
                out_features=["res2", "res3", "res4", "res5"],
            ).freeze(2),
            ["res2", "res3", "res4", "res5"],
            256,
            top_block=LastLevelMaxPool(),
        ),
        proposal_generator=RPN(
            in_features=["p2", "p3", "p4", "p5", "p6"],
            head=StandardRPNHead(in_channels=256, num_anchors=3),
            anchor_generator=DefaultAnchorGenerator(
                sizes=[[32], [64], [128], [256], [512]],
                aspect_ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
                offset=0.0,
            ),
            anchor_matcher=Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True),
            box2box_transform=Box2BoxTransform([1.0, 1.0, 1.0, 1.0]),
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_topk=(2000, 1000),
            post_nms_topk=(1000, 1000),
            nms_thresh=0.7,
        ),
        roi_heads=StandardROIHeads(
            num_classes=80,
            batch_size_per_image=512,
            positive_fraction=0.25,
            proposal_matcher=Matcher([0.5], [0, 1], allow_low_quality_matches=False),
            box_in_features=["p2", "p3", "p4", "p5"],
            box_pooler=ROIPooler(7, (1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32), 0, "ROIAlignV2"),
            box_head=FastRCNNConvFCHead(
                ShapeSpec(channels=256, height=7, width=7), conv_dims=[], fc_dims=[1024, 1024]
            ),
            box_predictor=FastRCNNOutputLayers(
                ShapeSpec(channels=1024),
                test_score_thresh=0.05,
                box2box_transform=Box2BoxTransform((10, 10, 5, 5)),
                num_classes=80,
            ),
            mask_in_features=["p2", "p3", "p4", "p5"],
            mask_pooler=ROIPooler(14, (1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32), 0, "ROIAlignV2"),
            mask_head=MaskRCNNConvUpsampleHead(
                ShapeSpec(channels=256, width=14, height=14),
                num_classes=80,
                conv_dims=[256, 256, 256, 256, 256],
            ),
        ),
        pixel_mean=[103.530, 116.280, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        input_format="BGR",
    )
    ```