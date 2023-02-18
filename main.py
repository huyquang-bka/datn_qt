from PyQt5 import QtCore, QtGui, QtWidgets
from main_app.uis.controllers.c_main_window import MainWindow

if __name__ == "__main__":
    import sys
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
