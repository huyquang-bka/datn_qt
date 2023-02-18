import os
import matplotlib.pyplot as plt
import numpy

count_dict = {"car": [10, 20, 30, 40, 50], "motor": [
    60, 70, 80, 90, 100], "time": ["1", "2", "3", "4", "5"]}


def dict_to_graph(count_dict, fp):
    plt.figure(figsize=(10, 10))
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
    plt.show()
    plt.savefig(f"resources/graphs/{name}.png")


if __name__ == "__main__":
    dict_to_graph(count_dict, "test.mp4")
