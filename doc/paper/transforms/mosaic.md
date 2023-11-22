# Mosaic

Mosaic数据增强方法是[YOLOV4](https://link.zhihu.com/?target=https%3A//github.com/Tianxiaomo/pytorch-YOLOv4)论文中提出来的，主要思想是将四张图片进行随机裁剪，再拼接到一张图上作为训练数据。这样做的好处是丰富了图片的背景，并且四张图片拼接在一起变相地提高了batch_size，在进行batch normalization的时候也会计算四张图片，所以对本身batch_size不是很依赖，单块GPU就可以训练YOLOV4。

- cutout和cutmix就是填充区域像素值的区别；
- mixup和cutmix是混合两种样本方式上的区别：
- mixup是将两张图按比例进行插值来混合样本，cutmix是采用cut部分区域再补丁的形式去混合图像，不会有图像混合后不自然的情形。