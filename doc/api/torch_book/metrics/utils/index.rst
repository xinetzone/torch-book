torch_book.metrics.utils
========================

.. py:module:: torch_book.metrics.utils


Classes
-------

.. autoapisummary::

   torch_book.metrics.utils.CropBorder


Module Contents
---------------

.. py:class:: CropBorder(size = 0, *args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   裁掉图像的边界

   Args:
       size: 图像每条边缘裁剪的像素。这些裁剪掉的像素不参与 PSNR 的计算。默认值为 0。


   .. py:method:: forward(x)


   .. py:attribute:: size


