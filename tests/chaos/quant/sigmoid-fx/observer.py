# from torch.nn import functional as F
import torch
from torch import fx
from torch.ao.quantization.observer import HistogramObserver

@fx.wrap
def histogram_observer(x):
    observer = HistogramObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    return observer(x)