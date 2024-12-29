from typing import Any
import torch
from detectron2.utils.events import EventStorage


def norms(Z):
    """计算除第一个维外的所有维的范数"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None]


def pgd_l2(model, X, epsilon, alpha, num_iter):
    """基于 PGD L2 构造样本 X 的对抗样本"""
    target_loss_idx = [0]
    losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]    
    delta = torch.zeros_like(X['image'], dtype=torch.float32, requires_grad=True)
    with EventStorage(0):
        for t in range(num_iter):
            X['image'] = X['image'] + delta
            losses = model([X])
            loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx])
            if t % 5 == 0:
                print(f"iter: {t}, loss: {loss}")            
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X['image']), 255-X['image']) # clip X+delta to [0,255]
            delta.data *= epsilon /  norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
    del losses
    return X, delta.detach()

def pgd_linf(model, X, epsilon=0.1, alpha=0.01,
             num_iter=20, randomize=False):
    """基于 PGD Linf 构造样本 X 的对抗样本"""
    target_loss_idx = [0]
    # 调整此损失函数以获得不同类型的结果(hallucination/miscclassification/misdetections)
    losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]
    if randomize:
        delta = torch.rand_like(X['image'], dtype=torch.float32, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X['image'], dtype=torch.float32, requires_grad=True)
    with EventStorage(0):
        for t in range(num_iter):
            X['image'] = X['image'] + delta
            losses = model([X])
            #print(losses)
            loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx])
            if t % 5 == 0:
                print(f"iter: {t}, loss: {loss}")
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
    del losses
    return X, delta.detach()
