# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\self_project\datn_qt\resources\uis\widget_graph.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WidgetGraph(object):
    def setupUi(self, WidgetGraph):
        WidgetGraph.setObjectName("WidgetGraph")
        WidgetGraph.resize(737, 611)
        self.gridLayout = QtWidgets.QGridLayout(WidgetGraph)
        self.gridLayout.setObjectName("gridLayout")
        self.label_graph = QtWidgets.QLabel(WidgetGraph)
        self.label_graph.setStyleSheet("background: black;\n"
"border: 25px;\n"
"border-radius: 2px solid black;")
        self.label_graph.setObjectName("label_graph")
        self.gridLayout.addWidget(self.label_graph, 1, 1, 1, 1)
        self.comboBox_list_graph = QtWidgets.QComboBox(WidgetGraph)
        self.comboBox_list_graph.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_list_graph.setStyleSheet("background: lightblue;\n"
"border: 10px;\n"
"border-radius: 10px solid black;")
        self.comboBox_list_graph.setObjectName("comboBox_list_graph")
        self.gridLayout.addWidget(self.comboBox_list_graph, 0, 1, 1, 1)

        self.retranslateUi(WidgetGraph)
        QtCore.QMetaObject.connectSlotsByName(WidgetGraph)

    def retranslateUi(self, WidgetGraph):
        _translate = QtCore.QCoreApplication.translate
        WidgetGraph.setWindowTitle(_translate("WidgetGraph", "Form"))
        self.label_graph.setText(_translate("WidgetGraph", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WidgetGraph = QtWidgets.QWidget()
    ui = Ui_WidgetGraph()
    ui.setupUi(WidgetGraph)
    WidgetGraph.show()
    sys.exit(app.exec_())