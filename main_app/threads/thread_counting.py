from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from ..util.detect_yolov5 import Tracking
from ..util.tools import count_object


class ThreadCounting(QThread):
    sig_car_count = pyqtSignal(str)
    sig_motor_count = pyqtSignal(str)

    def __init__(self, polygon, output_queue):
        super().__init__()
        self.__thread_active = False
        self.output_queue = output_queue
        self.polygon = polygon
        self.tracking = Tracking()

    def setup_fp(self, fp):
        self.fp = fp

    def convert_polygon(self, polygon):
        polygon = np.array(polygon, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        return polygon

    def slot_get_polygon(self, polygon):
        self.polygon = polygon

    def setup_model(self, model_path, device):
        self.tracking.weights = model_path
        self.tracking.device = device
        self.tracking._load_model()

    def run(self):
        self.__thread_active = True
        self.setup_model(
            r"resources\Weights\last_vehicle_23122022.pt", "0")
        cap = cv2.VideoCapture(self.fp)
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        old_id_dict = {}
        count_car = 0
        count_motor = 0
        while self.__thread_active:
            ret, frame = cap.read()
            if not ret:
                break
            polygon = []
            for point in self.polygon:
                polygon.append((int(point[0]*W), int(point[1]*H)))
            polygon = self.convert_polygon(polygon)
            id_dict = self.tracking.track(frame)
            for id, bbox in id_dict.items():
                x1, y1, x2, y2 = bbox[:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(id), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
            if self.output_queue.empty():
                self.output_queue.put(frame)
            num_car, num_motor = count_object(
                polygon, old_id_dict, id_dict)
            count_car += num_car
            count_motor += num_motor
            old_id_dict = id_dict.copy()
            self.sig_car_count.emit(str(count_car))
            self.sig_motor_count.emit(str(count_motor))
            self.msleep(1)

    def stop(self):
        self.__thread_active = False
