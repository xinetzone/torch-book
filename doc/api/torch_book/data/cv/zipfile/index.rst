torch_book.data.cv.zipfile
==========================

.. py:module:: torch_book.data.cv.zipfile


Classes
-------

.. autoapisummary::

   torch_book.data.cv.zipfile.LoadBufferFromZipFile


Module Contents
---------------

.. py:class:: LoadBufferFromZipFile

   从 `.zip` 文件中加载图片 buffer 列表


   .. py:method:: __call__(file_name)

      加载图片的二进制内容

      Args:
          file_name: zip 中图片的名称，例如：'0.jpg'
      Returns:
          buffer: 图片的二进制内容



   .. py:method:: __getitem__(index)

      加载图片的二进制内容

      Args:
          index: 图片的索引
      Returns:
          buffer: 图片的二进制内容



   .. py:method:: __len__()

      返回图片数量



   .. py:method:: __post_init__()


   .. py:attribute:: path
      :type:  str | pathlib.Path


