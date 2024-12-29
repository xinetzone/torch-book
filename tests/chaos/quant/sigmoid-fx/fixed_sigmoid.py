import logging
from sympy import interpolating_spline, symbols, exp
import torch
from torch import nn

def sigmoid(x):
    x = 1 + exp(-x)
    return 1/x

def get_nodes(indexes=[-6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6],
              func=sigmoid):
    """boundary 保证函数取得边界值"""
    x = symbols("x")
    if 0.0 not in indexes:
        indexes.append(0.0)
    indexes.sort() # 排序
    boundary = max(abs(indexes[-1]), abs(indexes[0]))
    indexes[-1] = boundary
    indexes[0] = - boundary
    # simplify(sigmoid(1) + sigmoid(-1))
    logging.debug(f"值的范围：\n {indexes}")
    sigmoid_sym = interpolating_spline(1,
                                    x,
                                    indexes,
                                    [float(func(n)) for n in indexes])
    nodes = {}
    for expr, set_pair in sigmoid_sym.as_expr_set_pairs():
        nodes[set_pair] = float(expr.diff()), float(expr.taylor_term(0, x))
    return nodes

def linear_sigmoid(x, nodes):
    y = torch.zeros_like(x)
    for set_pair, (scale, zero_point) in nodes.items():
        start, end = [torch.tensor(k, dtype=torch.float32) 
                      for k in set_pair.args[:2]]
        scale, zero_point = [torch.tensor(k, dtype=torch.float32) 
                             for k in [scale, zero_point]]
        cond = (x >= start) * (x < end)
        f = x * scale + zero_point
        y += f * cond
    y += torch.ones_like(x) * (x >= end)
    return y #* (y <=1) * (y >= 0)
