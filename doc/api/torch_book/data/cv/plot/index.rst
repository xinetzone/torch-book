torch_book.data.cv.plot
=======================

.. py:module:: torch_book.data.cv.plot


Classes
-------

.. autoapisummary::

   torch_book.data.cv.plot.CompareGridFrame
   torch_book.data.cv.plot.GridFrame


Module Contents
---------------

.. py:class:: CompareGridFrame

   对比显示两个图像列表的网格布局


   .. py:method:: __call__(frames1, frames2, **kwargs)

      横向堆叠两个图像的列表



   .. py:method:: __post_init__()

      初始化网格布局的图像和轴



   .. py:method:: update_axes(axes, frames)

      更新轴以显示图像



   .. py:attribute:: layout
      :type:  str
      :value: 'col'



   .. py:attribute:: num_cols
      :type:  int


   .. py:attribute:: num_rows
      :type:  int


   .. py:attribute:: scale
      :type:  float
      :value: 1.5



.. py:class:: GridFrame

   用于存储网格布局的图像和轴


   .. py:method:: __call__(frames, **kwargs)

      以网格布局绘制一系列图像的列表



   .. py:method:: __post_init__()

      初始化网格布局的图像和轴



   .. py:method:: update_axes(axes, frames)

      更新轴以显示图像



   .. py:attribute:: num_cols
      :type:  int


   .. py:attribute:: num_rows
      :type:  int


   .. py:attribute:: scale
      :type:  float
      :value: 1.5



