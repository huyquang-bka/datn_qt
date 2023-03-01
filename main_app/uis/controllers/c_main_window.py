import queue
from PyQt5 import QtCore, QtGui, QtWidgets
from ..views.main_window import Ui_MainWindow
from .c_widget_draw_polygon import WidgetDrawPolygon
from .c_widget_graph import WidgetGraph
from ...threads.thread_counting import ThreadCounting
import cv2


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.polygon = []
        self.polygon_speed = []
        self.fp = ""
        self.connect_buttons()

        self.create_queue()
        self.create_thread()

        self.widget_draw_polygon = WidgetDrawPolygon()
        self.widget_draw_polygon.sig_send_polygon.connect(
            self.slot_get_polygon)
        self.widget_draw_polygon_speed = WidgetDrawPolygon()
        self.widget_draw_polygon_speed.sig_send_polygon.connect(
            self.slot_get_polygon_speed)

        self.widget_graph = WidgetGraph()

        self.ui.btn_stop.setEnabled(False)

    def connect_buttons(self):
        self.ui.btn_choose_file.clicked.connect(self.choose_file)
        self.ui.btn_start.clicked.connect(self.start)
        self.ui.btn_stop.clicked.connect(self.stop)
        self.ui.btn_draw_polygon.clicked.connect(self.draw_polygon)
        self.ui.btn_draw_speed.clicked.connect(self.draw_speed)
        self.ui.btn_graph.clicked.connect(self.show_graph)

    def slot_get_polygon(self, polygon):
        self.polygon = polygon
        self.thread_counting.slot_get_polygon(polygon)

    def slot_get_polygon_speed(self, polygon):
        self.polygon_speed = polygon
        self.thread_counting.slot_get_polygon_speed(polygon)

    def create_queue(self):
        self.output_queue = queue.Queue()

    def create_thread(self):
        self.thread_counting = ThreadCounting(
            self.polygon, self.polygon_speed, self.output_queue)
        self.thread_counting.sig_car_count.connect(
            self.ui.qlabel_count_car.setText)
        self.thread_counting.sig_motor_count.connect(
            self.ui.qlabel_count_motor.setText)

    def choose_file(self):
        # only choose video files
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a video file", "", "Video Files (*.mp4 *.avi *.mkv, *.flv, *.MTS)")
        self.fp = fp
        if fp:
            self.polygon = []
            self.polygon_speed = []

    def paintEvent(self, e):
        if self.output_queue.qsize() > 0:
            current_frame = self.output_queue.get()
            self.show_frame(self.ui.qlabel_frame, current_frame)
        self.update()

    @ staticmethod
    def show_frame(frame_camera, current_frame):
        if current_frame is not None:
            rgb_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (1280, 720))
            qt_img = QtGui.QPixmap.fromImage(
                QtGui.QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QtGui.QImage.Format_RGB888)).scaled(
                frame_camera.width(), frame_camera.height())
            frame_camera.setPixmap(qt_img)
            frame_camera.setScaledContents(True)

    def start(self):
        if not self.polygon:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Vẽ vùng đếm trước khi bắt đầu")
            return
        if not self.polygon_speed:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Vẽ vùng đếm tốc độ trước khi bắt đầu")
            return
        try:
            foo = 1 / int(self.ui.qline_speed.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Nhập giá trị tốc độ hợp lệ")
            return
        self.ui.qlabel_count_car.setText("0")
        self.ui.qlabel_count_motor.setText("0")
        self.thread_counting.setup_fp(self.fp)
        self.thread_counting.distance = int(self.ui.qline_speed.text())
        self.thread_counting.start()
        self.ui.btn_stop.setEnabled(True)
        self.ui.btn_start.setEnabled(False)

    def stop(self):
        self.thread_counting.stop()
        self.ui.btn_start.setEnabled(True)
        self.ui.btn_stop.setEnabled(False)

    def draw_polygon(self):
        self.widget_draw_polygon.show_frame(self.fp)

    def draw_speed(self):
        self.widget_draw_polygon_speed.show_frame(self.fp)

    def show_graph(self):
        self.widget_graph.reload()
        self.widget_graph.show()
