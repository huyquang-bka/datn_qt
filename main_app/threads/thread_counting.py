from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from ..util.detect_yolov5 import Tracking
from ..util.tools import count_object
import matplotlib.pyplot as plt
import os


def convert_time_to_ms(seconds):
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    time_str = f"{minutes}:{seconds}"
    return time_str


def dict_to_graph(count_dict, fp):
    plt.figure(figsize=(8, 8))
    name = os.path.basename(fp)
    name = os.path.splitext(name)[0]
    car_count = count_dict["car"]
    motor_count = count_dict["motor"]
    max_type = car_count if car_count[-1] > motor_count[-1] else motor_count
    time = count_dict["time"]
    plt.plot(time, car_count, "ro-", label="Car")
    plt.plot(time, motor_count, "bx-", label="Motor")
    for i, v in enumerate(car_count):
        y = v + int(v / 30) + 1
        plt.text(i, y, "%d" % v, ha="center")
    for i, v in enumerate(motor_count):
        y = v + int(v / 30) + 1
        plt.text(i, y, "%d" % v, ha="center")
    plt.ylim(0, max_type[-1] + 25)
    plt.xlabel("Time (s)")
    plt.ylabel("Count")
    plt.legend(loc="upper left")
    plt.title(
        f"Counting result for {name}\nCar: {car_count[-1]}\nMotor: {motor_count[-1]}")
    plt.savefig(f"resources/graphs/{name}.png")


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
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        old_id_dict = {}
        self.count_car = 0
        self.count_motor = 0
        self.frame_count = 0
        self.graph_dict = {"car": [], "motor": [], "time": []}
        while self.__thread_active:
            ret, frame = cap.read()
            if not ret:
                self.graph_dict["car"].append(self.count_car)
                self.graph_dict["motor"].append(self.count_motor)
                self.graph_dict["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))
                dict_to_graph(self.graph_dict, self.fp)
                break
            self.frame_count += 1
            if self.frame_count % (5 * self.fps) == 0 and self.frame_count != 0:
                self.graph_dict["car"].append(self.count_car)
                self.graph_dict["motor"].append(self.count_motor)
                self.graph_dict["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))
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
            self.count_car += num_car
            self.count_motor += num_motor
            old_id_dict = id_dict.copy()
            self.sig_car_count.emit(str(self.count_car))
            self.sig_motor_count.emit(str(self.count_motor))
            self.msleep(1)

    def stop(self):
        self.graph_dict["car"].append(self.count_car)
        self.graph_dict["motor"].append(self.count_motor)
        self.graph_dict["time"].append(
            convert_time_to_ms(self.frame_count / self.fps))
        dict_to_graph(self.graph_dict, self.fp)
        self.__thread_active = False
