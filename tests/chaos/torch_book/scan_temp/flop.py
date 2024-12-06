from typing import Any, Sequence
import traceback
import torch
from torch.fx.passes.shape_prop import ShapeProp, _extract_tensor_metadata
from torch.fx.node import Argument, Node, Target, map_aggregate
from torch.fx._compatibility import compatibility
from .flop_ops import MODULE_FLOPs_MAPPING, METHOD_FLOPs_MAPPING, FUNCTION_FLOPs_MAPPING

class FLOP(ShapeProp):
    def __init__(self, gm, fake_mode=None, ignore_ops: Sequence[str] = []):
        super().__init__(gm, fake_mode=fake_mode)
        self.ignore_ops = ignore_ops
    @compatibility(is_backward_compatible=True)
    def call_module(self, target : 'Target', args : tuple[Argument, ...], kwargs : dict[str, Any]) -> Any:
        # 从环境中检索执行的 args 和 kwargs 值
        # 执行该方法并返回结果
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        result = submod(*args, **kwargs)
        # 计算出来 result 之后再计算FLOPs，保证计算过程能正确执行
        mod_name = submod.__class__.__name__
        flops = None
        if mod_name in MODULE_FLOPs_MAPPING:
            if mod_name not in self.ignore_ops:
                flops = MODULE_FLOPs_MAPPING[mod_name](submod, result, *args, **kwargs)
            else:
                flops = 0
        return result, flops

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result
        result = target(*args, **kwargs)

        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        func_name = target.__name__
        flops = None
        if func_name in FUNCTION_FLOPs_MAPPING:
            if func_name not in self.ignore_ops:
                flops = FUNCTION_FLOPs_MAPPING[func_name](result, *args, **kwargs)
            else:
                flops = 0

        return result, flops

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(target, str)
        result = getattr(self_obj, target)(*args_tail, **kwargs)

        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        method_name = target
        flops = None
        if method_name in METHOD_FLOPs_MAPPING:
            if method_name not in self.ignore_ops:
                flops = METHOD_FLOPs_MAPPING[method_name](self_obj, result, *args_tail, **kwargs)
            else:
                flops = 0
        return result, flops

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    raise ValueError("'fake_mode' must be None.")
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
                        n.meta['flops'] = flops
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

def get_FLOPs(gm, sample_input, ignore_ops=[]):
    FLOP(gm).propagate(sample_input)
    result_table = []
    for node in gm.graph.nodes:
        node: Node
        _result_row = [node.name, node.op, str(node.target)]

        node_module_name = ''
        if (_var_name := 'nn_module_stack') in node.meta:
            node_module_name = next(reversed(node.meta[_var_name].values())).__name__
            # node_module_name = ".".join([_v.__name__ for _v in node.meta[_var_name].values()])
        _result_row.append(node_module_name)

        if (_var_name := 'flops') in node.meta:
            flops = node.meta[_var_name]
            if flops is None:
                _result_row.append('not_recognized')
            elif isinstance(flops, int):
                if node_module_name in ignore_ops:
                    _result_row.append('ignored')
                else:
                    _result_row.append(flops)
            else:
                raise TypeError(type(flops))
        else:
            raise KeyError("'flops' must be in node.meta")

        result_table.append(_result_row)
    return result_table