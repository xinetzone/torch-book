from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

@dataclass
class GridFrame:
    """用于存储网格布局的图像和轴"""
    num_rows: int # 网格的行数
    num_cols: int # 网格的列数
    scale: float = 1.5 # 每个图像的大小比例

    def __post_init__(self):
        """初始化网格布局的图像和轴"""
        figsize = (self.num_cols * self.scale, self.num_rows * self.scale)
        self.figure = plt.figure(figsize=figsize, layout="constrained")

    def update_axes(self, axes, frames):
        """更新轴以显示图像"""
        for ax, img in zip(axes, frames):
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        return axes

    def __call__(self, frames, **kwargs):
        """以网格布局绘制一系列图像的列表"""
        gs = GridSpec(self.num_rows, self.num_cols, self.figure, **kwargs)
        axes = [self.figure.add_subplot(g) for g in gs]
        axes = self.update_axes(axes, frames)
        return gs, axes
    
@dataclass
class CompareGridFrame:
    """对比显示两个图像列表的网格布局"""
    num_rows: int # 网格的行数
    num_cols: int # 网格的列数
    scale: float = 1.5 # 每个图像的大小比例
    layout: str ="col" # 布局方式，'row' 或 'col'

    def __post_init__(self):
        """初始化网格布局的图像和轴"""
        if self.layout == "col":
            figsize = (self.num_cols * 2 * self.scale, self.num_rows * self.scale)
        elif self.layout == "row":
            figsize = (self.num_cols * self.scale, self.num_rows * 2 * self.scale)
        else:
            raise ValueError("layout must be 'col' or 'row'")
        self.figure = plt.figure(figsize=figsize, layout="constrained")

    def update_axes(self, axes, frames):
        """更新轴以显示图像"""
        for ax, img in zip(axes, frames):
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        return axes

    def __call__(self, frames1, frames2, **kwargs):
        """横向堆叠两个图像的列表"""
        if self.layout == "col":
            gs_main = GridSpec(1, 2, figure=self.figure, width_ratios=[1, 1], hspace=0.01)  # 宽度比例为 2:1
        elif self.layout == "row":
            gs_main = GridSpec(2, 1, figure=self.figure, height_ratios=[1, 1], wspace=0.01)  # 高度比例为 2:1
        else:
            raise ValueError("layout must be 'col' or 'row'")
        gs_left = GridSpecFromSubplotSpec(self.num_rows, self.num_cols, subplot_spec=gs_main[0], **kwargs)
        axes_left = [self.figure.add_subplot(g) for g in gs_left]
        left_axes = self.update_axes(axes_left, frames1)
        gs_right = GridSpecFromSubplotSpec(self.num_rows, self.num_cols, subplot_spec=gs_main[1], **kwargs)
        axes_right = [self.figure.add_subplot(g) for g in gs_right]
        right_axes = self.update_axes(axes_right, frames2)
        return left_axes, right_axes
