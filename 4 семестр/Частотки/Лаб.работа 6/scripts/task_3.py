import os

import cv2 as cv
import numpy as np

from .drawer import drawer


def fft(image, width, height):
    return np.fft.fft2(image, (height, width))


def ifft(image):
    return np.clip(np.fft.ifft2(image), 0, 1).real


def task(filename: str):
    os.makedirs('sources/3third', exist_ok=True)
    os.chdir('sources/3third')

    image = cv.imread(f'../{filename}', cv.IMREAD_GRAYSCALE) / 255
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpened = cv.filter2D(image, ddepth=-1, kernel=kernel).clip(0, 1)
    sharpened = cv.filter2D(sharpened, ddepth=-1, kernel=kernel).clip(0, 1)
    drawer(sharpened, 'sharpened')

    h, w = image.shape
    height, width = h + 2, w + 2  # h + 3 - 1, w + 3 - 1

    sharpened_fft = np.multiply(fft(image, width, height), fft(kernel, width, height))
    sharpened = ifft(sharpened_fft)
    sharpened_fft = np.multiply(fft(sharpened, width, height), fft(kernel, width, height))
    sharpened = ifft(sharpened_fft)
    drawer(sharpened, 'sharpened_fft')

    os.chdir('../..')
