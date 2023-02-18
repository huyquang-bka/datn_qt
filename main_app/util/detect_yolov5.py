import os
import time
import torch
from ..yolov5_module.models.common import DetectMultiBackend
from ..yolov5_module.utils.general import (
    check_img_size, non_max_suppression, scale_coords)
from ..yolov5_module.utils.augmentations import letterbox
import numpy as np
import cv2
from .sort import Sort


class Detection:
    def __init__(self):
        self.weights = 'resources/Weight/face_v3.pt'
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cpu'
        self.classes = None
        self.agnostic_nms = True
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def _load_model(self):
        # Load model
        # self.device = select_device(self.device)
        if self.device == "cpu":
            arg = "cpu"
        else:
            arg = f"cuda:{self.device}"
        print(self.weights, self.classes,
              self.conf_thres, arg, self.imgsz)
        self.device = torch.device(arg)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=self.dnn, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size

    def detect(self, image):
        image_copy = image.copy()
        bboxes = []
        im = letterbox(image, self.imgsz, stride=self.stride,
                       auto=self.pt)[0]  # resize
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
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], image_copy.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    bboxes.append([x1, y1, x2, y2, int(cls), float(conf)])

        return bboxes


class Tracking(Detection):
    def __init__(self, max_age=70, min_hits=3, iou_threshold=0.3):
        super().__init__()
        self._tracker = Sort(max_age=max_age, min_hits=min_hits,iou_threshold=iou_threshold)

    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self._tracker.update(dets_to_sort)
        if len(tracked_det) > 0:
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 8]
            categories = tracked_det[:, 4]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(
                    map(lambda x: max(0, int(x)), bbox_xyxy[i]))
                id = int(indentities[i])
                track_dict[id] = [x1, y1, x2, y2, categories[i]]
        return track_dict


if __name__ == "__main__":
    tracker = Tracking()

    tracker.weights = r"resources/Weight/face_v3.pt"
    tracker.imgsz = 320
    tracker.device = "cpu"
    tracker.conf_thres = 0.25
    tracker.classes = [1]
    tracker.agnostic_nms = True
    tracker.half = False
    tracker._load_model()

    # path = "rtsp://admin:Atin%402022@192.168.1.233/profile1/media.smp"
    path = 0
    cap = cv2.VideoCapture(path)
    count = 0
    old_time = 0
    while True:
        if time.time() - old_time > 1:
            print(count)
            count = 0
            old_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        count += 1
        t = time.time()
        id_dict = tracker.track(image)
        for id, bbox in id_dict.items():
            x1, y1, x2, y2, cls = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            print("Tracking id: ", id)
        image = cv2.resize(image, (640, 480))
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
