from dataclasses import dataclass
import torch
from torch import nn
from torch.optim import lr_scheduler
from ..backend.gpu import gpu, num_gpus as _num_gpus

to = lambda x, *args, **kwargs: x.to(*args, **kwargs)

@dataclass
class Trainer:
    max_epochs: int
    use_gpu_id: int = 0
    gradient_clip_val: int = 0

    def __post_init__(self):
        self.num_gpus = _num_gpus()

    # def prepare_scheduler(self, model,
    #                       lr=0.00142857,
    #                       momentum=0.857142,
    #                       weight_decay=0.00857142,
    #                       lr_period=4,
    #                       lr_decay=0.857142,
    #                       param_group=True,
    #                       **kwargs):
    #     if param_group==True: # 适用于微调
    #         params_1x = [param for name, param in model.net.named_parameters()
    #                      if name not in ["fc.weight", "fc.bias"]]
    #         params = [
    #             {'params': params_1x},
    #             {'params': model.net.fc.parameters(),
    #             'lr': lr * 10}
    #         ]
    #     else:
    #         params = model.parameters()
    #     self.optim = model.configure_optimizers(params,
    #                                             lr=lr,
    #                                             momentum=momentum,
    #                                             weight_decay=weight_decay,
    #                                             **kwargs)
    #     self.scheduler = lr_scheduler.StepLR(self.optim, 
    #                                          step_size=lr_period, 
    #                                          gamma=lr_decay)
        
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

    
