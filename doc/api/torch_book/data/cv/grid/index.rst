torch_book.data.cv.grid
=======================

.. py:module:: torch_book.data.cv.grid


Classes
-------

.. autoapisummary::

   torch_book.data.cv.grid.FlattenPairedGrid
   torch_book.data.cv.grid.Grid
   torch_book.data.cv.grid.GridConfig
   torch_book.data.cv.grid.PairedGrid
   torch_book.data.cv.grid.PairedRandomCrop


Module Contents
---------------

.. py:class:: FlattenPairedGrid(*args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   将 LR 和 HR 图像裁剪成 Grid 数据对展平


   .. py:method:: forward(lr, hr)


.. py:class:: Grid(*args: Any, device: Optional[torch._prims_common.DeviceLikeType] = None)
              Grid(storage)
              Grid(other)
              Grid(size, *, device = None)

   Bases: :py:obj:`torchvision.tv_tensors.TVTensor`


   Base class for all TVTensors.

   You probably don't want to use this class unless you're defining your own
   custom TVTensors. See
   :ref:`sphx_glr_auto_examples_transforms_plot_custom_tv_tensors.py` for details.


   .. py:method:: __len__()

      返回 Grid 中包含的图像块数量



   .. py:method:: flatten()

      将 Grid 数据展平为 (h*w, C, H, W) 形状的 Tensor



   .. py:method:: randmeshgrid(indexing = 'ij')

      返回随机排列的索引，用于打乱 Grid 中图像块的顺序



   .. py:method:: shuffle(indexes = None)

      随机打乱 Grid 中图像块的顺序



   .. py:method:: unflatten()

      将展平的 Grid 数据恢复为 (h, w, num_cols, C, H, W) 形状



.. py:class:: GridConfig

   网格的配置


   .. py:method:: __rshift__(scale)

      将网格配置按 `scale` 因子缩小



   .. py:method:: make_space(h, w)


   .. py:attribute:: crop_size
      :type:  int
      :value: 480



   .. py:attribute:: step
      :type:  int
      :value: 240



   .. py:attribute:: thresh_size
      :type:  int
      :value: 0



.. py:class:: PairedGrid(scale, config, *args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   将 LR 和 HR 图像裁剪成 Grid 数据对


   .. py:method:: forward(lr, hr)


   .. py:attribute:: config


   .. py:attribute:: scale


.. py:class:: PairedRandomCrop(scale, gt_patch_size, *args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   一种用于图像数据增强的技术，通常用于生成图像对（例如高分辨率图像和低分辨率图像）的训练数据。

   主要目的是确保在数据增强过程中，高分辨率图像和低分辨率图像的裁剪区域保持一致，从而保证训练数据的配对关系。


   .. py:method:: forward(lr, hr)


   .. py:attribute:: gt_patch_size


   .. py:attribute:: lq_patch_size


   .. py:attribute:: scale


