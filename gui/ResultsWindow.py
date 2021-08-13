from PyQt5.QtWidgets import (QLabel,
                             QMainWindow,
                             QFileDialog,
                             QPushButton,
                             QHBoxLayout,
                             QVBoxLayout,
                             QWidget,
                             QMessageBox)
from PyQt5.QtCore import Qt
from skimage.io import imsave
from image_processing import *


class ResultsWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)
        self.initUI()
        self.resImage = None

    def initUI(self):
        self.setWindowTitle("Results")
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        self.inputImage = self.createLabelImage()
        self.corruptedImage = self.createLabelImage()
        self.corruptedImage.setAlignment(Qt.AlignCenter)
        self.restoredImage = self.createLabelImage()

        imagesLayout = QHBoxLayout()
        imagesLayout.addWidget(self.inputImage)
        imagesLayout.addWidget(self.corruptedImage)
        imagesLayout.addWidget(self.restoredImage)

        inputImageLabel = QLabel("Input Image")
        inputImageLabel.setAlignment(Qt.AlignCenter)
        corruptedImageLabel = QLabel("Corrupted Image")
        corruptedImageLabel.setAlignment(Qt.AlignCenter)
        restoredImageLabel = QLabel("Restored Image")
        restoredImageLabel.setAlignment(Qt.AlignCenter)

        labelsLayout = QHBoxLayout()
        labelsLayout.addWidget(inputImageLabel)
        labelsLayout.addWidget(corruptedImageLabel)
        labelsLayout.addWidget(restoredImageLabel)

        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.onSaveButtonClicked)
        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.closeButton)

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(imagesLayout)
        self.mainLayout.addLayout(labelsLayout)
        self.mainLayout.addLayout(buttonLayout)
        self.mainWidget.setLayout(self.mainLayout)

        self.setFixedSize(self.minimumSize().width(), self.minimumSize().height())

    def createLabelImage(self):
        label = QLabel()
        label.setFixedSize(256, 256)
        return label

    def onSaveButtonClicked(self):
        name = QFileDialog.getSaveFileName(self, 'Save file')[0]
        if name != "":
            imsave(name, self.resImage)
            self.showMessage("Image was successfully saved!")

    def showMessage(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Image saving")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    def setImages(self, inputImage, corruptedImage, restoredImage):
        self.inputImage.setPixmap(image2pixmap(inputImage))
        self.corruptedImage.setPixmap(image2pixmap(corruptedImage))
        self.restoredImage.setPixmap(image2pixmap(restoredImage))
        self.resImage = restoredImage
