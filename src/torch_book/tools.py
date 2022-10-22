import torch
from torch import nn
from .runner import Accumulator, Timer
from .vision.mp_plot import Animator

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

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用 GPU 计算模型在数据集上的精度"""
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

def train(net, train_iter, test_iter, num_epochs, lr, device):
    """用 GPU 训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')