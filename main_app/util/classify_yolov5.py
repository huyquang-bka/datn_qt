import torch
import torch.nn.functional as F

from ..yolov5_module.models.common import DetectMultiBackend
from ..yolov5_module.utils.augmentations import classify_transforms
from ..yolov5_module.utils.general import check_img_size
import cv2
import os


class Classify:
    def __init__(self) -> None:
        self.weights = ""
        self.source = "",
        self.imgsz = (224, 224),
        self.device = '',
        self.half = False

    def _load_model(self):
        # Load model
        self.device = "cpu" if self.device == "cpu" else f"cuda:{self.device}"
        self.device = torch.device(self.device)
        print(self.weights, self.device, self.half, self.imgsz, self.source)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, fp16=self.half, dnn=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size
        self.transforms = classify_transforms(self.imgsz[0], half=self.half, device=self.device)

    def predict(self, img):
        im = self.transforms(img)
        im = torch.Tensor(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        results = self.model(im)

        pred = F.softmax(results, dim=1)  # probabilities
        for i, prob in enumerate(pred):  # per image
            # Print results
            top1i = prob.tolist()  # top 5 indices
            accuracy = max(top1i)
            cls = top1i.index(accuracy)
            return cls, accuracy


if __name__ == "__main__":
    classifier = Classify()
    classifier.weights = "resources/Weight/last.pt"
    classifier.imgsz = (224, 224)
    classifier.device = "0"
    classifier.half = False
    classifier._load_model()

    path = r"D:\Company\restaurant\resources\FaceLog"

    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, file_name))
        if img is None:
            continue
        result = classifier.predict(img)
        # cv2.putText(img, result, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow("img", img)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
