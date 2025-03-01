from dataclasses import dataclass
from typing import Any
from pathlib import Path
import logging
import torch
from torch import nn
from taolib.protocol.timer import Timer
# from taolib.protocol.animator import Animator
from taolib.plot.animator import Animator
from .utils import Accumulator

logger = logging.getLogger(__name__)

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


def accuracy(y_hat, y):
    """计算正确预测的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter, ctx=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not ctx:
            ctx = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(ctx) for x in X]
            else:
                X = X.to(ctx)
            y = y.to(ctx)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


@dataclass
class Classifier:
    mod: nn.Module
    loss: Any
    optimizer: Any
    scheduler: Any
    train_iter: Any
    test_iter: Any
    ctx: Any
    timer: Timer

    def __post_init__(self):
        logger.info(f'training on {self.ctx}')
        self.num_batches = len(self.train_iter)
        # def init_weights(m):
        #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
        #         nn.init.xavier_uniform_(m.weight)
        # self.mod.apply(init_weights)
        self.mod.to(self.ctx)
        
    def fit(self, num_epochs, checkpoint_dir=None, checkpoint_interval=1, resume_from=None, start_epoch = 0):
        """训练模型"""
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.mod.state_dict(), checkpoint_dir/f'best_model_params.pth')
        self.animator = Animator(
            xlabel='epoch', xlim=[start_epoch, start_epoch+num_epochs], ylim=[0, 1],
            legend=['train loss', 'train acc', 'test acc']
        )
        # 断点恢复逻辑
        
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=torch.device(self.ctx))
            self.mod.load_state_dict(checkpoint)
            logger.info(f"Resumed training from epoch {start_epoch}")
        best_acc = 0.0
        for epoch in range(start_epoch, num_epochs+start_epoch):
            # 训练损失之和，训练准确率之和，样本数
            metric = Accumulator(3)
            self.mod.train()
            for i, (X, y) in enumerate(self.train_iter):
                self.timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.ctx), y.to(self.ctx)
                y_hat = self.mod(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                self.timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (self.num_batches // 5) == 0 or i == self.num_batches - 1:
                    self.animator.add(epoch + (i + 1) / self.num_batches,
                                      (train_l, train_acc, None))
            test_acc = evaluate_accuracy(self.mod, self.test_iter, ctx=self.ctx)
            
            logger.debug(f"epoch {epoch:05d}: ctx {self.ctx}, loss {train_l:.5g}, train acc {train_acc:.5g}, test acc {test_acc:.5g}")
            self.animator.add(epoch + 1, (None, None, test_acc))
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            # 保存检查点
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': self.mod.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None
                }
                path = checkpoint_dir/f'epoch_{epoch+1}.pth'
                torch.save(checkpoint, path)
                logger.debug(f"Saved checkpoint to {path}")
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(self.mod.state_dict(), checkpoint_dir/f'best_model_params.pth')
        logger.info(
            f"loss {train_l:.5g}, train acc {train_acc:.5g}, test acc {test_acc:.5g}")
        logger.info(
            f"{metric[2] * num_epochs / self.timer.sum():.5g} examples/sec on {str(self.ctx)}")
