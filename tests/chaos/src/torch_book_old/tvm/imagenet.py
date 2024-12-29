from PIL import Image
import numpy as np
from tvm.contrib.download import download_testdata
from ..data.imagenet import Transforms


class TestImage:
    def __init__(self, img_url, name):
        # img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        self.img_path = download_testdata(img_url, name, module="data")
        self.im = Image.open(self.img_path)
        

    def get_imagenet_input(self, im_height, im_width):
        transforms = Transforms((im_height, im_width))
        im = self.im.resize((im_height, im_width))
        preprocess = transforms.test
        pt_tensor = preprocess(im)
        return np.expand_dims(pt_tensor.numpy(), 0)


def get_synset():
    """获取 ImageNet 标签
    """
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())
