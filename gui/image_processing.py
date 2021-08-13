from skimage import color, transform
from PyQt5 import QtGui
import numpy as np


def add_noise(image, m, std):
    noise = np.random.normal(m, std, image.shape)
    noise = noise.reshape(image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def image2pixmap(image):
    height, width, channels = image.shape
    bytesPerLine = channels * width
    qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap(qImg)
    return pixmap


def resize_image(image, shape):
    res = transform.resize(image, shape)
    res = (res * 255).astype(np.uint8)
    return res

def image2tensor(image):
    image = (image - 127.5) / 127.5
    image = image[np.newaxis, ...]
    return image


def output2image(output):
    image = (output[0] + 127.5) * 127.5
    image = image.astype(np.uint8)
    return image


def lab_output2image(L, ab):
    image = np.zeros(shape=(256, 256, 3), dtype=np.float32)
    image[:, :, 0] = (L[:, :, 0] + 1.) * 50
    image[:, :, 1:] = ab[0, :, :, :] * 128
    image = color.lab2rgb(image) * 255
    image = image.astype(np.uint8)
    return image