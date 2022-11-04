import torch
from torch import nn
from torch.optim import SGD
from .utils import HyperParameters
from .plotx import ProgressBoard


def _cpu():
    return torch.device('cpu')


_to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
_numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)


class Module(nn.Module, HyperParameters):
    def __init__(self, plot_train_per_epoch=2,
                 plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, _numpy(_to(value, _cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self, 
                             lr=0.00142857,
                             momentum=0.857142,
                             weight_decay=0.00857142,
                             **kwargs):
        """默认情况下，使用随机梯度下降优化器。
        """
        return SGD(self.parameters(), 
                   lr=lr,
                   momentum=momentum,
                   dampening=0,
                   weight_decay=weight_decay,
                   **kwargs)

    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
