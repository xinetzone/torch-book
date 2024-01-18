torch_book.scan.modules
========================

The modules subpackage contains tools for inspection of modules.

.. currentmodule:: torch_book.scan.modules


FLOPs
-----
Related to the number of floating point operations performed during model inference.

.. autofunction:: torch_book.scan.modules.flops.module_flops


MACs
-----
Related to the number of multiply-accumulate operations performed during model inference

.. autofunction:: torch_book.scan.modules.macs.module_macs


DMAs
----
Related to the number of direct memory accesses during model inference

.. autofunction:: torch_book.scan.modules.memory.module_dmas


Receptive field
---------------
Related to the effective receptive field of a layer

.. autofunction:: torch_book.scan.modules.receptive.module_rf
