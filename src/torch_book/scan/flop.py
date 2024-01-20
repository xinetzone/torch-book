import numpy as np
from typing import Any
from torch import Tensor, nn
from torch.types import Number
from torch.fx._compatibility import compatibility
from .common import FLOPsABC


class ElementwiseFLOPs(FLOPsABC):
    @compatibility(is_backward_compatible=True)
    def fetch_method_flops(self, self_obj: Tensor, result: Tensor, *args_tail, **kwargs):
        """计算方法的FLOPs"""
        return np.prod(result.shape)

    @compatibility(is_backward_compatible=True)
    def fetch_function_flops(self, result: Tensor|Number, *args, **kwargs) -> Any:
        """计算函数的FLOPs"""
        assert len(args) == 2, len(args)
        total_flops = None
        if isinstance(result, Number):
            total_flops = 1
        elif isinstance(result, Tensor):
            total_flops = np.prod(result.shape)
        else:
            raise TypeError(type(result))
        return total_flops

    @compatibility(is_backward_compatible=True)
    def fetch_module_flops(self, module: nn.Module, result: Tensor, *args, **kwargs) -> Any:
        """计算模块的FLOPs"""
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        assert isinstance(result, Tensor)
        input_shape = args[0].shape  # [..., d_in]
        result_shape = result.shape
        assert input_shape == result_shape
        total_flops = np.prod(result_shape)
        return total_flops
