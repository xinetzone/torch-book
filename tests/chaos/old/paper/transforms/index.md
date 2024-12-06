# 数据增强

一些常用的数据增强方法：

- Cutout：通过 0 填充，随机删除矩形区域
- Random Erasing：通过均值填充，随机删除矩形区域，
- Mixup：两张图像每个位置的像素根据一定比例进行叠加，label 根据像素叠加比例进行分配
- Cutmix：随机删除矩形区域，并通过另一张图像的同一位置像素值填充，label 根据像素所占比例进行分配


```{toctree}
cutout
random-erasing
mixup
cutmix
mosaic
```