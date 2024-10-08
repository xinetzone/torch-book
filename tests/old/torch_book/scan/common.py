"""Common utilities"""
from abc import ABC, abstractmethod
from typing import Any
from torch.fx.passes.shape_prop import ShapeProp, _extract_tensor_metadata
from torch.fx.node import Node, map_aggregate
import torch
import traceback
from torch.fx.node import Argument, Node
from torch.fx._compatibility import compatibility


class FLOPsABC(ShapeProp, ABC):
    # def __init__(self, gm, fake_mode=None):
    #     super().__init__(gm, fake_mode)
    #     self._bunch
    @compatibility(is_backward_compatible=True)
    @abstractmethod
    def fetch_method_flops(self, self_obj: Any, result: Any, *args_tail, **kwargs):
        """计算方法的FLOPs"""
        ...

    @compatibility(is_backward_compatible=True)
    @abstractmethod
    def fetch_function_flops(self, result: Any, *args, **kwargs) -> Any:
        """计算函数的FLOPs"""
        ...

    @compatibility(is_backward_compatible=True)
    @abstractmethod
    def fetch_module_flops(self, module: Any, result: Any, *args, **kwargs) -> Any:
        """计算模块的FLOPs"""
        ...
    
    @compatibility(is_backward_compatible=True)
    def call_method(self, target : 'Target', args : tuple[Argument, ...], kwargs : dict[str, Any]) -> Any:
        # args[0] 是此方法调用的 `self` 对象
        self_obj, *args_tail = args
        # 执行方法并返回结果
        assert isinstance(target, str)
        result = getattr(self_obj, target)(*args_tail, **kwargs)
        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        flops = self.fetch_method_flops(self_obj, result, *args_tail, **kwargs)
        return result, flops
        
    @compatibility(is_backward_compatible=True)
    def call_function(self, target : 'Target', args : tuple[Argument, ...], kwargs : dict[str, Any]) -> Any:
        assert not isinstance(target, str)
        # 执行函数并返回结果
        result = target(*args, **kwargs)
        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        # func_name = target.__name__
        # 调用子类的方法计算 FLOPs
        flops = self.fetch_function_flops(result, *args, **kwargs)
        return result, flops
    
    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        # 从环境中检索执行的 args 和 kwargs 值
        # 执行该方法并返回结果
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        result = submod(*args, **kwargs)
        # 计算出来 result 之后再计算FLOPs，保证计算过程能正确执行
        # mod_name = submod.__class__.__name__
        # 调用子类的方法计算 FLOPs
        flops = self.fetch_module_flops(submod, result, *args, **kwargs)
        return result, flops

    def run_node(self, n : Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    # with self.fake_mode, enable_python_dispatcher():
                    #     result = super().run_node(n)
                    raise ValueError("'fake_mode' 暂未支持.")
                else:
                    with self._set_current_node(n):
                        args, kwargs = self.fetch_args_kwargs_from_env(n)
                        assert isinstance(args, tuple)
                        assert isinstance(kwargs, dict)
                        if n.op in ('call_module', 'call_function', 'call_method'):
                            result, flops = getattr(self, n.op)(n.target, args, kwargs)
                        else:
                            result = getattr(self, n.op)(n.target, args, kwargs)
                            flops = 0
                        assert flops not in n.meta, n.meta.keys()
                        n.meta['FLOPs'] = flops
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)
        return result
