import time
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from ..util.detect_yolov5 import Tracking
from ..util.tools import count_object, is_in_polygon
import matplotlib.pyplot as plt
import os


def convert_time_to_ms(seconds):
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    time_str = f"{minutes}:{seconds}"
    return time_str


def dict_to_graph(count_dict, fp, speed=False, stay=False):
    plt.figure(figsize=(20, 20))
    name = os.path.basename(fp)
    title = name
    if speed:
        name = os.path.splitext(name)[0] + "_speed"
    elif stay:
        name = os.path.splitext(name)[0] + "_stop"
    else:
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
    if speed:
        plt.ylabel("Speed (km/h)")
    elif stay:
        plt.ylabel("Time waiting (s)")
    else:
        plt.ylabel("Counting")
    plt.legend(loc="upper left")
    if speed:
        avarage_speed_car = np.mean(car_count)
        avarage_speed_motor = np.mean(motor_count)
        plt.title(
            f"Speed result for {title}\nCar avarage speed: {avarage_speed_car:.2f} km/h\nMotor avarage speed: {avarage_speed_motor:.2f} km/h")
    elif stay:
        average_time_car = np.mean(car_count)
        average_time_motor = np.mean(motor_count)
        plt.title(
            f"Time waiting result for {title}\nCar avarage time: {average_time_car:.2f} s\nMotor avarage time: {average_time_motor:.2f} s")
    else:
        plt.title(
            f"Counting result for {title}\nCar: {car_count[-1]}\nMotor: {motor_count[-1]}")
    plt.savefig(f"resources/graphs/{name}.png")


class ThreadCounting(QThread):
    sig_car_count = pyqtSignal(str)
    sig_motor_count = pyqtSignal(str)

    def __init__(self, polygon, polygon_speed, output_queue):
        super().__init__()
        self.__thread_active = False
        self.output_queue = output_queue
        self.polygon = polygon
        self.polygon_speed = polygon_speed
        self.distance = 0
        self.tracking = Tracking()

    def setup_fp(self, fp):
        self.fp = fp

    def convert_polygon(self, polygon):
        polygon = np.array(polygon, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        return polygon

    def slot_get_polygon(self, polygon):
        self.polygon = polygon

    def slot_get_polygon_speed(self, polygon):
        self.polygon_speed = polygon

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
        self.graph_dict_speed = {"car": [], "motor": [], "time": []}
        self.graph_dict_stay = {"car": [], "motor": [], "time": []}
        self.speed_dict = {}
        self.save_speed_dict = {}
        self.stay_dict = {}
        self.save_stay_dict = {}
        count = 0
        old_time = time.time()
        fps_show = 0
        while self.__thread_active:
            if time.time() - old_time > 1:
                fps_show = round(count / (time.time() - old_time))
                count = 0
                old_time = time.time()
            ret, frame = cap.read()
            if not ret:
                self.graph_dict["car"].append(self.count_car)
                self.graph_dict["motor"].append(self.count_motor)
                self.graph_dict["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))
                dict_to_graph(self.graph_dict, self.fp)

                self.graph_dict_speed["car"].append(
                    int(np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 0] + [0])))
                self.graph_dict_speed["motor"].append(
                    (np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 1] + [0])))
                self.graph_dict_speed["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))
                dict_to_graph(self.graph_dict_speed,
                              f"{self.fp}", speed=True)

                self.graph_dict_stay["car"].append(
                    int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 0] + [0])))
                self.graph_dict_stay["motor"].append(
                    int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 1] + [0])))
                self.graph_dict_stay["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))
                dict_to_graph(self.graph_dict_stay,
                              f"{self.fp}", stay=True)

                break
            self.frame_count += 1
            count += 1
            if self.frame_count % (5 * self.fps) == 0 and self.frame_count != 0:
                self.graph_dict["car"].append(self.count_car)
                self.graph_dict["motor"].append(self.count_motor)
                self.graph_dict["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))

                self.graph_dict_speed["car"].append(
                    int(np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 0] + [0])))
                self.graph_dict_speed["motor"].append(
                    int(np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 1] + [0])))
                self.graph_dict_speed["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))

                self.graph_dict_stay["car"].append(
                    int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 0] + [0])))
                self.graph_dict_stay["motor"].append(
                    int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 1] + [0])))
                self.graph_dict_stay["time"].append(
                    convert_time_to_ms(self.frame_count / self.fps))

            polygon = []
            polygon_speed = []
            for point in self.polygon:
                polygon.append((int(point[0]*W), int(point[1]*H)))
            for point in self.polygon_speed:
                polygon_speed.append((int(point[0]*W), int(point[1]*H)))
            polygon = self.convert_polygon(polygon)
            polygon_speed = self.convert_polygon(polygon_speed)
            id_dict = self.tracking.track(frame)
            for id, bbox in id_dict.items():
                x1, y1, x2, y2, cls = bbox[:5]
                if is_in_polygon([x2, y2], polygon_speed):
                    try:
                        self.speed_dict[id] += 1
                    except:
                        self.speed_dict[id] = 1
                else:
                    if id in self.speed_dict:
                        list_speed = [self.distance /
                                      (self.speed_dict[id] / self.fps) * 3.6, cls]
                        if 3 < list_speed[0] <= 60:
                            self.save_speed_dict[id] = list_speed
                        del self.speed_dict[id]
                if is_in_polygon([x2, y2], polygon):
                    try:
                        self.stay_dict[id] += 1
                    except:
                        self.stay_dict[id] = 1
                else:
                    if id in self.stay_dict:
                        list_stay = [self.stay_dict[id] / self.fps, cls]
                        self.save_stay_dict[id] = list_stay
                        del self.stay_dict[id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if id in self.save_speed_dict:
                    cv2.putText(frame, str(round(self.save_speed_dict[id][0])) + " km/h", (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
            cv2.polylines(frame, [polygon_speed], True, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {fps_show}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

        self.graph_dict_speed["car"].append(
            int(np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 0] + [0])))
        self.graph_dict_speed["motor"].append(
            int(np.mean([v[0] for k, v in self.save_speed_dict.items() if int(v[1]) == 1] + [0])))
        self.graph_dict_speed["time"].append(
            convert_time_to_ms(self.frame_count / self.fps))

        self.graph_dict_stay["car"].append(
            int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 0] + [0])))
        self.graph_dict_stay["motor"].append(
            int(np.mean([v[0] for k, v in self.save_stay_dict.items() if int(v[1]) == 1] + [0])))
        self.graph_dict_stay["time"].append(
            convert_time_to_ms(self.frame_count / self.fps))
        dict_to_graph(self.graph_dict, self.fp)
        dict_to_graph(self.graph_dict_speed, f"{self.fp}", speed=True)
        dict_to_graph(self.graph_dict_stay, f"{self.fp}", stay=True)
        self.__thread_active = False
