import sys
from PyQt5.QtWidgets import QApplication
from OptionsWindow import OptionsWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptionsWindow()
    window.show()
    app.exec_()
