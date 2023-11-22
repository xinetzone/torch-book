from dataclasses import dataclass
from typing import Any
import torch
from pathlib import Path
from .api.tool.utils import *
from .api.tool.torch_utils import *
from .api.tool.darknet2pytorch import Darknet

from ...base import DetectorABC

ROOT = Path(__file__).resolve().parent

@dataclass
class Yolov4(DetectorABC):
    input_tensor_size: int = 416
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name: str = "yolov4"
    ori_size: tuple[int] = (416, 416)
    conf_thres: int = 0.5
    iou_thres: int = 0.45
    max_labels: int = 20
    
    def __post_init__(self):
        self.test: int = 0

    def requires_grad_(self, state: bool):
        assert self.detector
        self.detector.models.requires_grad_(state)

    def load(self, model_weights, cfg_type=""):
        if cfg_type: # "shakedrop", "tiny"
            config_file = f"{ROOT}/configs/yolov4-{cfg_type}.cfg"
        else:
            config_file = f"{ROOT}/configs/yolov4.cfg"
        self.detector = Darknet(config_file).to(self.device)

        self.detector.load_weights(model_weights)
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)

        bbox_array = post_processing(batch_tensor, self.conf_thres, self.iou_thres, detections_with_grad)
        for i, pred in enumerate(bbox_array):
            pred = torch.Tensor(pred).to(self.device)
            if len(pred) != 0:
                pred[:, :4] = torch.clamp(pred[:, :4], min=0, max=1)
            bbox_array[i] = pred # shape([1, 6])
        # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
        # v4's confs is the combination of obj conf & cls conf
        confs = detections_with_grad[1]
        # print(confs.shape)
        # cls_max_ids = torch.argmax(confs, dim=2)
        # print(cls_max_ids.shape)
        max_confs = torch.max(confs, dim=2)[0]
        output = {'bbox_array': bbox_array, 'obj_confs': max_confs, "cls_max_ids": None}
        return output
