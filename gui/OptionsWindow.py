from PyQt5.QtWidgets import (QLabel,
                             QMainWindow,
                             QFileDialog,
                             QPushButton,
                             QHBoxLayout,
                             QVBoxLayout,
                             QWidget,
                             QRadioButton,
                             QCheckBox,
                             QGridLayout,
                             QMessageBox)
from PyQt5.QtCore import Qt
import skimage
from ResultsWindow import ResultsWindow
from GAN import *


class OptionsWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)

        self.imagePath = ""
        self.currentOperation = ""
        self.noiseAdded = False
        self.resizedBefore = False

        self.initUI()
        self.resultsWindow = ResultsWindow()
        self.colorizator = ColorizationGAN("../GANs/colorization")
        self.denoisator = DenoisingGAN("../GANs/denoising")
        self.srgan = SRGAN("../GANs/super_resolution")


    def initUI(self):
        self.setWindowTitle("Image Restorer")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.openImageButton = QPushButton("Browse...")
        self.openImageButton.clicked.connect(self.openImage)
        self.openImageLabel = QLabel("Open image")
        self.openImageLayout = QHBoxLayout()
        self.openImageLayout.addWidget(self.openImageLabel)
        self.openImageLayout.addWidget(self.openImageButton)
        self.openImageLayout.addStretch(1)
        self.colorizeRadioButton = QRadioButton("Colorize")
        self.colorizeRadioButton.operation = "colorization"
        self.colorizeRadioButton.toggled.connect(self.onRadioButtonClicked)

        self.denoiseRadioButton = QRadioButton("Denoise")
        self.denoiseRadioButton.operation = "denoising"
        self.denoiseRadioButton.toggled.connect(self.onRadioButtonClicked)

        self.enlargeRadioButton = QRadioButton("Enlarge")
        self.enlargeRadioButton.operation = "enlarging"
        self.enlargeRadioButton.toggled.connect(self.onRadioButtonClicked)

        self.addNoiseCheckBox = QCheckBox("Add Noise")
        self.addNoiseCheckBox.operation = "noise"
        self.addNoiseCheckBox.setEnabled(False)

        self.optionsGridLayout = QGridLayout()
        self.optionsGridLayout.addWidget(self.colorizeRadioButton, 0, 0)
        self.optionsGridLayout.addWidget(self.denoiseRadioButton, 1, 0)
        self.optionsGridLayout.addWidget(self.enlargeRadioButton, 2, 0)
        self.optionsGridLayout.addWidget(self.addNoiseCheckBox, 1, 1)


        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.onStartButtonClicked)
        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.startButton)
        self.buttonLayout.addWidget(self.closeButton)
        self.buttonLayout.addStretch(1)

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.openImageLayout)
        self.mainLayout.addLayout(self.optionsGridLayout)
        self.mainLayout.addLayout(self.buttonLayout)
        self.mainWidget.setLayout(self.mainLayout)

        self.setFixedSize(self.minimumSize().width(), self.minimumSize().height())

    def openImage(self):
        self.showMessage("Image will be resized to 256x256 pixels")
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'd:\\', "Image files (*.jpg *.png *.jpeg *.JPEG)")
        self.imagePath = fname[0]

    def showMessage(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def onRadioButtonClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.currentOperation = radioButton.operation
            if radioButton.operation == "denoising":
                self.addNoiseCheckBox.setEnabled(True)
            else:
                self.addNoiseCheckBox.setEnabled(False)

    def onStartButtonClicked(self):
        if self.imagePath == "":
            self.showMessage("Image wasn't selected!")
            return
        if self.currentOperation == "":
            self.showMessage("Choose operation type!")
            return
        trueImage = skimage.io.imread(self.imagePath)
        trueImage = resize_image(trueImage, (256, 256))
        if trueImage.ndim != 3:
            trueImage = skimage.color.gray2rgb(trueImage)
            trueImage = (trueImage * 255).astype(np.uint8)
        if trueImage.shape[2] == 4:
            trueImage = skimage.color.rgba2rgb(trueImage)
            trueImage = (trueImage * 255).astype(np.uint8)

        if self.currentOperation == "colorization":
            inputImage = skimage.color.rgb2lab(trueImage)
            inputImage = inputImage[:, :, 0]
            inputImage = inputImage[..., np.newaxis]
            resultImage = self.colorizator.process(inputImage)
            inputImage = skimage.color.gray2rgb(skimage.color.rgb2gray(trueImage)) * 255

        elif self.currentOperation == "denoising":
            if self.addNoiseCheckBox.isChecked():
                inputImage = add_noise(trueImage, 0, 10)
            else:
                inputImage = trueImage.copy()
            resultImage = self.denoisator.process(inputImage)

        elif self.currentOperation == "enlarging":
            inputImage = resize_image(trueImage, (128, 128))
            resultImage = self.srgan.process(inputImage)
            pass

        inputImage = inputImage.astype(np.uint8)
        inputImage = inputImage.astype(np.uint8)

        self.resultsWindow.setImages(trueImage, inputImage, resultImage)
        self.resultsWindow.setWindowModality(Qt.ApplicationModal)
        self.resultsWindow.show()