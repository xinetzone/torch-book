import torch
from torch.nn import functional as F
from .model import Module

float32 = torch.float32
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

@dataclass
class Classifier2(ModelABC):
    mod: nn.Module
    device = d2l.try_gpu()

    def __post_init__(self):
        #print('training on', self.device)
        # self.mod.apply(init_weights)
        self.mod.to(self.device)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs) # 用于延迟初始化
        if init is not None:
            self.mod.apply(init)

    def train_step(self, *args, **kwargs):
        ...

    def valid_step(self, *args, **kwargs):
        ...

    def configure(self, params,
                  lr=0.00142857, # 0.9
                  momentum=0.857142,
                  weight_decay=0.00857142, **kwargs):
        opt_kwargs = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        return SGD(params, **opt_kwargs)

    def fit(self, model, data_iter, max_epochs):
        self.prepare_data(data_iter)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(max_epochs):
            self.fit_epoch()


    def fit_epoch(self):
        self.mod.train()
        for batch in self.train_dataloader:
            loss = self.mod.train_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch
    

    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm