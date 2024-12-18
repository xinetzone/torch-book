torch_book.datasets.cv.div2k
============================

.. py:module:: torch_book.datasets.cv.div2k


Classes
-------

.. autoapisummary::

   torch_book.datasets.cv.div2k.PairedDIV2K


Module Contents
---------------

.. py:class:: PairedDIV2K

   成对图片数据集


   .. py:method:: __getitem__(index)

      加载(LR, HR)图片对

      Args:
          index: 图片的索引
      Returns:
          buffer: 图片的二进制内容



   .. py:method:: __len__()

      返回图片对数量



   .. py:method:: __post_init__()


   .. py:method:: _check()

      检查图片对是否匹配



   .. py:attribute:: HR_path
      :type:  str | pathlib.Path


   .. py:attribute:: LR_path
      :type:  str | pathlib.Path


   .. py:attribute:: scale
      :type:  int


