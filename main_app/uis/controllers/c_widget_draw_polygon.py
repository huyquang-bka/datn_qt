from PyQt5 import QtCore, QtGui, QtWidgets
from ..views.widget_draw_polygon import Ui_WidgetDrawPolygon
import cv2


class WidgetDrawPolygon(QtWidgets.QWidget):
    sig_send_polygon = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.ui = Ui_WidgetDrawPolygon()
        self.ui.setupUi(self)
        self.polygon = []
        self.ui.qlabel_frame = DrawLabel()
        self.ui.qlabel_frame.setStyleSheet("background: black")
        self.ui.qlabel_frame.setObjectName("qlabel_frame")
        self.ui.gridLayout.addWidget(self.ui.qlabel_frame, 0, 1, 1, 1)
        self.connect_buttons()

    def connect_buttons(self):
        self.ui.btn_save.clicked.connect(self.save)
        self.ui.btn_clear.clicked.connect(self.clear)

    def save(self):
        polygon_raito = []
        for point in self.ui.qlabel_frame.polygon:
            polygon_raito.append(
                (point[0] / self.ui.qlabel_frame.width(),
                 point[1] / self.ui.qlabel_frame.height()))
        self.sig_send_polygon.emit(polygon_raito)
        self.close()

    def clear(self):
        self.ui.qlabel_frame.polygon = []

    def show_frame(self, fp):
        if not fp:
            QtWidgets.QMessageBox.warning(self, "Warning", "Chọn file video")
            return
        cap = cv2.VideoCapture(fp)
        frame = cap.read()[1]
        self.ui.qlabel_frame.set_frame(frame)
        self.show()


class DrawLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.polygon = []
        self.setMouseTracking(True)
        self.setScaledContents(True)

    def set_frame(self, frame):
        self.frame = frame

    def mousePressEvent(self, event):
        # Add the clicked point to the polygon list
        self.polygon.append((event.pos().x(), event.pos().y()))
        # Redraw the label to show the newly added point
        self.update()

    def paintEvent(self, event):
        # Draw the frame image on the label
        painter = QtGui.QPainter(self)
        if self.frame is not None:
            current_frame = self.frame
            self.show_frame(painter, current_frame)
        # Draw the polygon on the label
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 5)
        painter.setPen(pen)
        for point in self.polygon:
            painter.drawPoint(point[0], point[1])
        if len(self.polygon) > 0:
            prev_point = self.polygon[0]
            for point in self.polygon:
                painter.drawLine(
                    prev_point[0], prev_point[1], point[0], point[1])
                prev_point = point
            painter.drawLine(
                prev_point[0], prev_point[1], self.polygon[0][0], self.polygon[0][1])
        self.update()

    def show_frame(self, painter, current_frame):
        if current_frame is not None:
            rgb_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (1280, 720))
            qt_img = QtGui.QPixmap.fromImage(
                QtGui.QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QtGui.QImage.Format_RGB888)).scaled(
                self.width(), self.height())
            painter.drawPixmap(self.rect(), qt_img)
