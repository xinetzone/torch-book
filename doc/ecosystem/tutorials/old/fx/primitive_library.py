import torch
from torch import fx

def sigmoid_lowp(x: torch.Tensor):
    x = x.float()
    x = x.sigmoid()
    return x.half()

fx.wrap(sigmoid_lowp)

def add_lowp(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float(), b.float()
    c = a + b
    return c.half()

torch.fx.wrap(add_lowp)

if __name__ == "__main__":
    # 看看在使用这些函数的代码中进行符号跟踪时会发生什么
    class Foo(torch.nn.Module):
        def forward(self, x, y):
            x = sigmoid_lowp(x)
            y = sigmoid_lowp(y)
            return add_lowp(x, y)


    traced = fx.symbolic_trace(Foo())
    print(traced.code)
    """
    def forward(self, x, y):
        sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
        sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
        add_lowp = __main___add_lowp(sigmoid_lowp, sigmoid_lowp_1);  sigmoid_lowp = sigmoid_lowp_1 = None
        return add_lowp
    """
    # 注意 `sigmoid_lowp` 和 `add_lowp` 的调用出现在跟踪中;他们自身没有被追踪