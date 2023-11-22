from dataclasses import dataclass
from typing import Any
from pathlib import Path
import torch
from .api.pytorchyolo.models import load_model
from .api.pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from ...base import DetectorABC

ROOT = Path(__file__).resolve().parent

@dataclass
class Yolov3(DetectorABC):
    input_tensor_size: int = 412
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name: str = "yolov3"
    ori_size: tuple[int] = (416, 416)
    conf_thres: int = 0.5
    iou_thres: int = 0.45
    max_labels: int = 20

    def requires_grad_(self, state: bool):
        self.detector.module_list.requires_grad_(state)
    
    def load(self, model_weights, cfg_type=""):
        if cfg_type: # "shakedrop", "tiny"
            config_file = f"{ROOT}/configs/yolov3-{cfg_type}.cfg"
        else:
            config_file = f"{ROOT}/configs/yolov3.cfg"
        self.detector = load_model(model_path=config_file, weights_path=model_weights).to(self.device)
        self.eval()

    def __call__(self, batch_tensor: torch.tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        obj_confs = detections_with_grad[:, :, 4]
        cls_max_ids = detections_with_grad[:, :, 5]

        bbox_array = []
        for i, pred in enumerate(preds):
            box = rescale_boxes(pred, self.input_tensor_size, self.ori_size)
            box[:, [0, 2]] /= self.ori_size[1]
            box[:, [1, 3]] /= self.ori_size[0]
            box[:, :4] = torch.clamp(box[:, :4], min=0, max=1)
            # print(box)
            bbox_array.append(box)

        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output