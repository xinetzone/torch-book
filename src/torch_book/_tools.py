import torch
from torch import nn
from .runner import Accumulator, Timer
from .plotx import Animator


class Fx:
    ones = torch.ones
    zeros = torch.zeros
    tensor = torch.tensor
    arange = torch.arange
    meshgrid = torch.meshgrid
    sin = torch.sin
    sinh = torch.sinh
    cos = torch.cos
    cosh = torch.cosh
    tanh = torch.tanh
    linspace = torch.linspace
    exp = torch.exp
    log = torch.log
    normal = torch.normal
    rand = torch.rand
    matmul = torch.matmul
    int32 = torch.int32
    float32 = torch.float32
    concat = torch.cat
    stack = torch.stack
    abs = torch.abs
    eye = torch.eye
    numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
    size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
    reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
    to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.
    """
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def accuracy(y_hat, y):
    """计算正确预测的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = Fx.argmax(y_hat, axis=1)
    cmp = Fx.astype(y_hat, y.dtype) == y
    return float(Fx.reduce_sum(Fx.astype(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter, device=None):
    """计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, valid_iter, 
          num_epochs, devices,
          optimizer, scheduler):
    """用 GPU 训练模型"""
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)
    # net.apply(init_weights)
    print('training on', devices)
    if torch.device('cpu') not in devices:
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    # scheduler = torch.optim.lr_scheduler.StepLR(trainer, 
    #                                             step_size=lr_period, 
    #                                             gamma=lr_decay)
    # loss = nn.CrossEntropyLoss()
    # https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html
    loss = nn.CrossEntropyLoss(reduction="none") # ”LogSumExp技巧” 
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels,
                                 loss, optimizer, devices)
            
            with torch.no_grad():
                metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = evaluate_accuracy(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
        measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')