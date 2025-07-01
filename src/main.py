from PyQt5.QtWidgets import QApplication
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()