from torch.nn import functional as F
from torch import float32, randn
from .model import Module
from .runner import Accumulator

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)


class Classifier(Module):
    def validation_step(self, batch):
        """报告验证批处理上的损失值和分类精度。
        
        为每个 `num_val_batches` 批绘制更新。
        这样做的好处是在整个验证数据上生成平均损失和准确性。
        
        如果最后一批包含的示例更少，这些平均数字就不是完全正确的，
        但是为了保持代码简单，我们忽略了这个微小的差异。
        """
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
        
        给定预测概率分布 Y_hat，当必须输出 hard 预测时，
        通常选择具有最高预测概率的类。
        
        当预测与标签类 Y 一致时，它们是正确的。
        分类精度是所有预测中正确的比例。
        
        尽管直接优化精度是很困难的(它是不可微的)，
        但它通常是我们最关心的性能度量。它通常是基准中的相关量。
        因此，在训练分类器时，几乎总是报告它。
        
        计算方法：
            首先，如果 `Y_hat` 是矩阵，假设第二维度存储每个类的预测分数。
            使用 `argmax` 通过每行中最大条目的索引获得预测类。
            然后将预测的类与 ground-truthy 的 Y 元素进行比较。
            由于 `==` 对数据类型非常敏感，
            因此将 `Y_hat` 的数据类型转换为与 `Y` 的数据类型匹配。
            结果是包含0(假)和1(真)项的张量。
            把它们加起来就能得到正确预测的次数。
        """
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return F.cross_entropy(Y_hat, Y, 
                               reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        X = randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = reshape(y, out.shape)
        l = loss(out, y)
        metric.add(reduce_sum(l), size(l))
    return metric[0] / metric[1]

