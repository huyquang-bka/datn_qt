import os
from PyQt5 import QtCore, QtGui, QtWidgets
from ..views.widget_graph import Ui_WidgetGraph


class WidgetGraph(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_WidgetGraph()
        self.ui.setupUi(self)
        self.graph_folder_path = "resources/graphs"
        self.connect_signals()

    def reload(self):
        self.ui.comboBox_list_graph.clear()
        self.ui.comboBox_list_graph.addItems(os.listdir(self.graph_folder_path))
        current_graph = self.ui.comboBox_list_graph.currentText()
        fp = os.path.join(self.graph_folder_path, current_graph)
        self.ui.label_graph.setPixmap(QtGui.QPixmap(fp))

    def connect_signals(self):
        self.ui.comboBox_list_graph.currentIndexChanged.connect(
            self.change_graph)

    def change_graph(self, index):
        current_graph = self.ui.comboBox_list_graph.currentText()
        fp = os.path.join(self.graph_folder_path, current_graph)
        self.ui.label_graph.setScaledContents(True)
        self.ui.label_graph.setPixmap(QtGui.QPixmap(fp))
        
        