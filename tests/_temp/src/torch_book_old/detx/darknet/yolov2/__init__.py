from dataclasses import dataclass
from typing import Any
from pathlib import Path
import torch
from .api.darknet import Darknet
from .api.utils import get_region_boxes, inter_nms
from ...base import DetectorABC

ROOT = Path(__file__).resolve().parent

@dataclass
class Yolov2(DetectorABC):
    input_tensor_size: int = 412
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name: str = "yolov2"
    ori_size: tuple[int] = (416, 416)
    conf_thres: int = 0.5
    iou_thres: int = 0.45
    max_labels: int = 20

    def load(self, model_weights, config_file=f"{ROOT}/configs/yolo.cfg"):
        self.detector = Darknet(config_file).to(self.device)
        self.detector.load_weights(model_weights)
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        # torch.tensor([1, num, classes_num+4+1])
        detections_with_grad = self.detector(batch_tensor)
        # x1, y1, x2, y2, det_conf, cls_max_conf, cls_max_id
        all_boxes, obj_confs, cls_max_ids = get_region_boxes(detections_with_grad, self.conf_thres,
                                                             self.detector.num_classes, self.detector.anchors,
                                                             self.detector.num_anchors)
        # print(all_boxes[0])
        all_boxes = inter_nms(
            all_boxes, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # print(all_boxes[0])
        obj_confs = obj_confs.view(batch_tensor.size(0), -1)
        cls_max_ids = cls_max_ids.view(batch_tensor.size(0), -1)
        bbox_array = []
        for boxes in all_boxes:
            # boxes = torch.cuda.FloatTensor(boxes)
            # pad_size = self.max_n_labels - len(boxes)
            # boxes = F.pad(boxes, (0, 0, 0, pad_size), value=0).unsqueeze(0)
            if len(boxes):
                boxes[:, :4] = torch.clamp(boxes[:, :4], min=0., max=1.)
            # print(boxes.shape)
            bbox_array.append(boxes)
            # bbox_array = torch.vstack((bbox_array, boxes)) if bbox_array is not None else boxes
        # print(bbox_array)

        output = {'bbox_array': bbox_array,
                  'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output

    def detect_test(self, batch_tensor):
        detections_with_grad = self.detector(batch_tensor)
        return detections_with_grad
