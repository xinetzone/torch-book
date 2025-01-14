# 使用内置数据集

数据集可以通过访问 {class}`detectron2.data.DatasetCatalog` 获取数据，或访问 {class}`detectron2.data.MetadataCatalog` 获取元数据(类名等)来使用。

本文档解释了如何设置内置数据集，以便它们可以被上述 API 使用。[](./datasets) 给出了如何使用 {class}`detectron2.data.DatasetCatalog` 和 {class}`detectron2.data.MetadataCatalog`，以及如何向它们添加新数据集的更深入的探讨。

Detectron2 内置对一些数据集的支持。假定数据集存在于环境变量 `DETECTRON2_DATASETS` 指定的目录中。如果需要，detectron2 将在这个目录下查找下面描述的结构中的数据集。

```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  cityscapes/
  VOC20{07,12}/
```

您可以通过 `export DETECTRON2_DATASETS=/path/to/datasets` 来设置内置数据集的位置。如果不设置，默认是相对于当前工作目录的 `./datasets`。

[MODEL ZOO](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) 包含使用这些内置数据集的配置和模型。

## 用于COCO 实例/关键点检测的预期数据集结构

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

你也可以使用 2014 年版本的数据集。

一些内置测试(`dev/run_*_tests.sh`)使用了 COCO 数据集的小版本，可以通过 `./datasets/prepare_for_tests.sh` 下载。

## PanopticFPN 的预期数据集结构

将 [COCO 网站](https://cocodataset.org/#download)上的全景(panoptic)注解提取为以下结构:

```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/  # png annotations
  panoptic_stuff_{train,val}2017/  # generated by the script mentioned below
```

安装 `panopticapi`:

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

然后，运行 `python datasets/prepare_panoptic_fpn.py`，从 panoptic 注解中提取语义。

## [LVIS 实例分割](https://www.lvisdataset.org/dataset) 的预期数据集结构

```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_image_info_test.json
  lvis_v1_{train,val}.json
  lvis_v1_image_info_test{,_challenge}.json
```

通过以下方式安装 `lvis-api`:

```bash
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
要评估使用 LVIS 注解在 COCO 数据集上训练的模型，请运行 `python datasets/prepare_cocofied_lvis.py` 来准备“cocofied” LVIS 注解。

## [城市景观](https://www.cityscapes-dataset.com/downloads/) 的预期数据集结构

```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```

安装城市景观脚本:

```bash
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

````{note}
注意: 要创建 `labelTrainIds.png`，首先准备上面的结构，然后运行 `cityscescript`:
```bash
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
````

实例分割不需要这些文件。

注意: 要生成城市景观全景数据集，运行 `cityscesescript`:
```bash
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
语义和实例分割不需要这些文件。

## [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) 的预期数据集结构

```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```

## [ADE20k 场景解析](http://sceneparsing.csail.mit.edu/)的预期数据集结构

```
ADEChallengeData2016/
  annotations/
  annotations_detectron2/
  images/
  objectInfo150.txt
```

目录 `annotations_detectron2` 是通过运行 `python datasets/ prepare_ade20k_sem_segg.py` 生成的。