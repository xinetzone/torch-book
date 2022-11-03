from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils import data
from torch.ao.quantization import disable_observer
from torch.ao.quantization.quantize import convert, prepare_qat
from torch.ao.quantization.qconfig import get_default_qat_qconfig
from torchvision import transforms
import torchvision

from runner import Timer, Accumulator, Animator
from .visual.mp_plot import show_images



class ModuleTool:
    '''将 inputs 转换为 NumPy 格式

    Args:
        inputs: 批量数据
        mean: 默认为 ImageNet 的 mean
        std: 默认为 ImageNet 的 std
        channel: 取值范围为 ['first', 'last']
    '''

    def __init__(self,
                 inputs: Tensor,
                 channel: str = 'first',
                 mean: list[float] = [0.485, 0.456, 0.406],
                 std: list[float] = [0.229, 0.224, 0.225]):
        self.inputs = inputs
        self.channel = channel
        self.mean, self.std = [mean, std]

    @property
    def images(self):
        inputs = self.inputs.cpu().numpy()
        if self.channel == 'first':
            inputs = inputs.transpose(0, 2, 3, 1)
        mean, std = (np.array(x) for x in [self.mean, self.std])
        inputs = std * inputs + mean
        inputs = np.clip(inputs, 0, 1)
        return inputs

    def outputs(self, model, inputs):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        return preds

    def imshow(self, model,
               class_names, device, *,
               num_rows=2, num_cols=2,
               scale=1.5):
        model.to(device)
        inputs = self.inputs.to(device)
        model.eval()
        with torch.no_grad():
            preds = self.outputs(model, inputs)
            titles = [class_names[x] for x in preds]
        return show_images(self.images,
                           num_rows, num_cols,
                           titles=titles, scale=scale)


class CV:
    @staticmethod
    def get_dataloader_workers():
        """Use 4 processes to read the data.
        """
        return 4

    @staticmethod
    def accuracy(y_hat, y):
        """Compute the number of correct predictions.

        """
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = Fx.argmax(y_hat, axis=1)
        cmp = Fx.astype(y_hat, y.dtype) == y
        return float(Fx.reduce_sum(Fx.astype(cmp, y.dtype)))

    @staticmethod
    def evaluate_accuracy(net, data_iter, device='cpu'):
        """计算在指定数据集上模型的精度

        """
        net = net.to(device)
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                X = X.to(device)
                y = y.to(device)
                metric.add(CV.accuracy(net(X), y), Fx.size(y))
        return metric[0] / metric[1]

    @staticmethod
    def evaluate_accuracy_gpu(net, data_iter, device=None):
        """Compute the accuracy for a model on a dataset using a GPU.

        Defined in :numref:`sec_lenet`"""
        if isinstance(net, nn.Module):
            net.eval()  # Set the model to evaluation mode
            if not device:
                device = next(iter(net.parameters())).device
        # No. of correct predictions, no. of predictions
        metric = Accumulator(2)

        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # Required for BERT Fine-tuning (to be covered later)
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(CV.accuracy(net(X), y), Fx.size(y))
        return metric[0] / metric[1]

    @staticmethod
    def train_batch(net, X, y, loss, trainer, device):
        """Train for a minibatch with mutiple GPUs.
        """
        if isinstance(X, list):
            # Required for BERT fine-tuning (to be covered later)
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        net.train()
        trainer.zero_grad()
        pred = net(X)
        l = loss(pred, y)
        l.sum().backward()
        trainer.step()
        train_loss_sum = l.sum()
        train_acc_sum = CV.accuracy(pred, y)
        return train_loss_sum, train_acc_sum

    @staticmethod
    def train(net, train_iter, test_iter,
              loss, trainer, num_epochs,
              device='cpu',
              need_prepare=False,
              is_freeze=False,
              is_quantized_acc=False,
              backend='fbgemm',
              ylim=[0, 1]):
        """Train a model with mutiple GPUs.
        """
        timer, num_batches = Timer(), len(train_iter)
        _ylim = '' if ylim[0] == 0 else f'{ylim[0]}+'
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=ylim,
                            legend=[f'{_ylim}train loss', 'train acc', 'test acc'])
        # nn.DataParallel(net, device_ids=devices).to(devices[0])
        net = net.to(device)
        if need_prepare:
            net.fuse_model()
            net.qconfig = get_default_qat_qconfig(backend)
            net = prepare_qat(net)
        for epoch in range(num_epochs):
            metric = Accumulator(4)
            if is_freeze:
                if epoch > 3:
                    # 冻结 quantizer 参数
                    net.apply(disable_observer)
                if epoch > 2:
                    # 冻结 batch 的平均值和方差估计
                    net.apply(nn.intrinsic.qat.freeze_bn_stats)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                l, acc = CV.train_batch(net, features,
                                        labels, loss,
                                        trainer, device)
                metric.add(l, acc, labels.shape[0], labels.numel())
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    # print((metric[0] / metric[2])+ylim[0])
                    animator.add(epoch + (i + 1) / num_batches,
                                 ((metric[0] / metric[2])+ylim[0], metric[1] / metric[3],
                                 None))
            if is_quantized_acc:
                quantized_model = deepcopy(net).to('cpu').eval()
                quantized_model = convert(quantized_model, inplace=False)
                test_acc = CV.evaluate_accuracy(quantized_model, test_iter)
            else:
                test_acc = CV.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))

        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(device)}')

    @staticmethod
    def train_fine_tuning(net,
                          train_iter, test_iter,
                          learning_rate,
                          num_epochs=5,
                          device='cuda:0',
                          is_freeze=False,
                          is_quantized_acc=False,
                          need_prepare=False,
                          param_group=True,
                          ylim=[0, 1],
                          output_layer='classifier'):
        # 如果param_group=True，输出层中的模型参数将使用十倍的学习率
        # param_name 可能为 'fc' 或者 'classifier'
        loss = nn.CrossEntropyLoss(reduction="none")
        if param_group:
            params_1x = [param for name, param in net.named_parameters()
                         if name.split('.')[0] != output_layer]
            trainer = torch.optim.SGD([{'params': params_1x},
                                       {'params': getattr(net, output_layer).parameters(),
                                        'lr': learning_rate * 10}],
                                      lr=learning_rate, weight_decay=0.001)
        else:
            trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                      weight_decay=0.001)
        CV.train(net, train_iter, test_iter,
                 loss, trainer, num_epochs,
                 device, ylim=ylim,
                 need_prepare=need_prepare,
                 is_freeze=is_freeze,
                 is_quantized_acc=is_quantized_acc)
