import torch
from torch import nn
from torch.optim import lr_scheduler
from .utils import HyperParameters
from .backend.gpu import gpu, num_gpus as _num_gpus


_to = lambda x, *args, **kwargs: x.to(*args, **kwargs)

class Trainer:


    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_batch(self, batch):
        if self.use_gpu_id in range(0, self.num_gpus):
            batch = [_to(a, gpu(self.use_gpu_id)) for a in batch]
        return batch
    
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.use_gpu_id in range(0, self.num_gpus):
            model.to(gpu(self.use_gpu_id))
        self.model = model
        
    def prepare_scheduler(self, model,
                          lr=0.00142857,
                          momentum=0.857142,
                          weight_decay=0.00857142,
                          lr_period=4,
                          lr_decay=0.857142,
                          param_group=True,
                          **kwargs):
        if param_group==True: # 适用于微调
            params_1x = [param for name, param in model.net.named_parameters()
                         if name not in ["fc.weight", "fc.bias"]]
            params = [
                {'params': params_1x},
                {'params': model.net.fc.parameters(),
                'lr': lr * 10}
            ]
        else:
            params = self.model.net.parameters()
        self.optim = model.configure_optimizers(params,
                                                lr=lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay,
                                                **kwargs)
        self.scheduler = lr_scheduler.StepLR(self.optim, 
                                             step_size=lr_period, 
                                             gamma=lr_decay)
        
    def fit(self, model, data,
            lr=0.00142857,
            momentum=0.857142,
            weight_decay=0.00857142,
            lr_period=4,
            lr_decay=0.857142,
            **kwargs):
        self.prepare_data(data)
        self.prepare_model(model)
        self.prepare_scheduler(model,
                               lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay,
                               lr_period=lr_period,
                               lr_decay=lr_decay,
                               **kwargs)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
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
        self.scheduler.step()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm


def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`subsec_gd-learningrate`"""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
