import torch

def cpu():
    return torch.device('cpu')

def num_gpus():
    """查询可用 GPU 的数量"""
    return torch.cuda.device_count()


def gpu(i=0):
    """Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    return [gpu(i) for i in range(num_gpus())]
