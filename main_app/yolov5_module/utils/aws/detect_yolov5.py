import os
import time
import torch

from ..models.common import DetectMultiBackend
from ..utils.general import (check_img_size, non_max_suppression, scale_coords)
from ..utils.torch_utils import select_device
from ..utils.augmentations import letterbox
import numpy as np
import cv2
from vehicle.utils.sort import *


class Detection:
    def __init__(self):
        self.weights = 'yolov5s.pt'
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = '0'
        self.classes = None
        self.agnostic_nms = False
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def _load_model(self):
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def detect(self, image):
        image_copy = image.copy()
        bboxes = []
        im = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]  # resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image_copy.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    bboxes.append([x1, y1, x2, y2, int(cls), float(conf)])

        return bboxes


class Tracking(Detection):
    def __init__(self):
        super().__init__()
        self._tracker = Sort(max_age=70, min_hits=0, iou_threshold=0.3)

    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self._tracker.update(dets_to_sort)
        if len(tracked_det) > 0:
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 8]
            categories = tracked_det[:, 4]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(map(int, bbox_xyxy[i]))
                id = int(indentities[i])
                track_dict[id] = [x1, y1, x2, y2, categories[i]]
        return track_dict


if __name__ == "__main__":
    path = r"C:\Users\ntdki\Downloads\plate"
    images = os.listdir(path)
    np.random.shuffle(images)

    tracker = Detection()

    tracker.weights = r"C:\Users\ntdki\Downloads\weights\digit_last_640.pt"
    tracker.imgsz = 640
    tracker.device = "cpu"
    tracker.conf_thres = 0.25
    tracker.classes = None
    tracker.agnostic_nms = True
    tracker.half = False

    tracker._load_model()

    for image_name in images:
        name, ext = os.path.splitext(image_name)
        if ext not in [".jpg", ".png"]:
            continue
        print(image_name)
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bboxes = tracker.detect(image)
        for bbox in bboxes:
            x1, y1, x2, y2, cls, conf = bbox
            print(cls, conf)
        print("*" * 20)
        image = cv2.resize(image, None, fx=5, fy=5)
        cv2.imshow(f"{cls}", image)
        key = cv2.waitKey(0)
        if key == 27:
            break
            