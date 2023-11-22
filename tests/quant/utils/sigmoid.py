from sympy import interpolating_spline, symbols, exp #, simplify
import torch
from torch.nn

def sigmoid(x):
    x = 1 + exp(-x)
    return 1/x

x = symbols("x")
# simplify(sigmoid(1) + sigmoid(-1))
node_indexes = [-6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6]
sigmoid_sym = interpolating_spline(1,
                                   x,
                                   node_indexes,
                                   [float(sigmoid(n)) for n in node_indexes])

nodes = {}
for expr, set_pair in sigmoid_sym.as_expr_set_pairs():
    nodes[set_pair] = float(expr.diff()), float(expr.taylor_term(0, x))
    
class Sigmoid(nn.Module):
    def forward(self, x):
        y = torch.zeros_like(x)
        for set_pair, (scale, zero_point) in nodes.items():
            start, end = [torch.tensor(k, dtype=torch.float32) 
                          for k in set_pair.args[:2]]
            scale, zero_point = [torch.tensor(k, dtype=torch.float32) 
                                 for k in [scale, zero_point]]
            cond = (x >= start) * (x < end)
            f = x * scale + zero_point
            y += f * cond
        return y