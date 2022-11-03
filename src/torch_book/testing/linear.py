import torch
from torch import nn
from ..model import *


class LinearRegressionScratch(Module):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = zeros(1, requires_grad=True)

    def forward(self, X):
        """The linear regression model.
    
        Defined in :numref:`sec_linear_scratch`"""
        return matmul(X, self.w) + self.b


    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_scratch`"""
        l = (y_hat - reshape(y, y_hat.shape)) ** 2 / 2
        return reduce_mean(l)


    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_scratch`"""
        return SGD([self.w, self.b], self.lr)


class SGD(HyperParameters):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad


    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class LinearRegression(Module):
    """Defined in :numref:`sec_linear_concise`"""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        """The linear regression model.
    
        Defined in :numref:`sec_linear_concise`"""
        return self.net(X)


    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_concise`"""
        fn = nn.MSELoss()
        return fn(y_hat, y)


    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_concise`"""
        return torch.optim.SGD(self.parameters(), self.lr)


    def get_w_b(self):
        """Defined in :numref:`sec_linear_concise`"""
        return (self.net.weight.data, self.net.bias.data)

