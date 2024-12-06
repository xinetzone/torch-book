from PIL import Image
import cv2
from IPython.display import display
import torch
from detectron2.utils.events import EventStorage


def attack_dt2(model:torch.nn.Module, input:dict, attack_fn=None, **kwargs)->tuple:
    """ Attack wrapper for DT2.  Takes pre-trained model, dt2 formattedinput, and runs 
    the specified attack with the supplied attack params passed as kwargs. 
    
    Parameters:
    model (torn.nn.Module): a pre-trained model compatible with DT2
    input (dict): a DT2-formatted input dict used as an input to the model
    attack_fn (Any): an attack method (see pgd_l2 and pgd_linf)

    Returns:
    tuple: the input containing perturbed image, a tensor representing the perturbation

    """
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True
    input, perturbation = attack_fn(model, input, kwargs['epsilon'], kwargs['alpha'], kwargs['num_iter'])
    model.training = False
    model.proposal_generator.training = False
    model.roi_heads.training = False
    return input, perturbation

def pgd_linf(model, X, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct PGD Linf adversarial examples on the examples X"""
    target_loss_idx = [0]
    losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]  ### Adjust this loss function to get different types of results (hallucination/miscclassification/misdetections)
    if randomize:
        delta = torch.rand_like(X['image'], dtype=torch.float32, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X['image'], dtype=torch.float32, requires_grad=True)
    with EventStorage(0) as storage:
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

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None]

def pgd_l2(model, X, epsilon, alpha, num_iter):
    """ Construct PGD L2 adversarial examples on the examples X"""
    target_loss_idx = [0]
    losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]    
    delta = torch.zeros_like(X['image'], dtype=torch.float32, requires_grad=True)
    with EventStorage(0) as storage:
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

def cv2_imshow(a):
  """A replacement for cv2.imshow() for use in Jupyter notebooks.

  Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
  """
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display(Image.fromarray(a))