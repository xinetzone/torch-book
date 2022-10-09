import os
import torch


class Result:
    def __init__(self, model):
        self.model = model

    def size(self):
        '''获取模型大小 (MB)
        '''
        torch.save(self.model.state_dict(), "tmp.pt")
        _size = os.path.getsize("tmp.pt")/(2**20)
        os.remove('tmp.pt')
        return _size
