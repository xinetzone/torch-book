from dataclasses import dataclass
from typing import Any
import logging
import time
import numpy as np
# from ..abcx.model import ModelABC
from torch import nn
from torch.optim import SGD, lr_scheduler
from torch import no_grad
from .utils import Accumulator
from ..backend.gpu import try_gpu
from ..plotx import Animator

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


def accuracy(y_hat, y):
    """计算正确预测的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


@dataclass
class Classifier:
    mod: nn.Module
    train_iter: Any
    test_iter: Any
    device_id: int = 0

    def __post_init__(self):
        self.device = try_gpu(self.device_id)
        logging.info(f'training on {self.device}')
        self.num_batches = len(self.train_iter)
        self.timer = Timer()
        self.loss = nn.CrossEntropyLoss()
        self.mod.to(self.device)

    def prepare_optimizer(self, lr=0.857142, momentum=0,
                          weight_decay=0, nesterov=False, **kwargs):
        params = self.mod.parameters() # 普通配置
        # 微调配置
        # params_1x = [param for name, param in model.net.named_parameters()
        #                  if name not in ["fc.weight", "fc.bias"]]
        # params = [
        #     {'params': params_1x},
        #     {'params': model.net.fc.parameters(),
        #     'lr': lr * 10}
        # ]
        self.optimizer = SGD(params, lr=lr, momentum=momentum,
                             weight_decay=weight_decay, nesterov=nesterov, **kwargs)
    
    def prepare_scheduler(self,
                          lr_period=4,
                          lr_decay=0.857142,
                          **kwargs):
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                             step_size=lr_period, 
                                             gamma=lr_decay,
                                             **kwargs)

    def prepare_animator(self, num_epochs):
        self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                                 legend=['train loss', 'train acc', 'test acc'])

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = Accumulator(3)
            self.mod.train()
            for i, (X, y) in enumerate(self.train_iter):
                self.timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.mod(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                with no_grad():
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                self.timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (self.num_batches // 5) == 0 or i == self.num_batches - 1:
                    self.animator.add(epoch + (i + 1) / self.num_batches,
                                      (train_l, train_acc, None))
            test_acc = evaluate_accuracy(self.mod, self.test_iter)
            self.animator.add(epoch + 1, (None, None, test_acc))
            if hasattr(self, "scheduler"):
                self.scheduler.step()
        logging.info(
            f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")
        logging.info(
            f"{metric[2] * num_epochs / self.timer.sum():.1f} examples/sec on {str(self.device)}")
