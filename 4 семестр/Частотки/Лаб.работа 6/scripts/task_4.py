import os

import cv2 as cv
import numpy as np

from .drawer import drawer


def fft(image, width, height):
    return np.fft.fft2(image, (height, width))


def ifft(image):
    return np.clip(np.fft.ifft2(image), 0, 1).real


def task(filename: str):
    os.makedirs('sources/4fourth', exist_ok=True)
    os.chdir('sources/4fourth')

    image = cv.imread(f'../{filename}', cv.IMREAD_GRAYSCALE) / 255
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    edged = cv.filter2D(image, ddepth=-1, kernel=kernel).clip(0, 1)
    drawer(edged, 'edged')

    h, w = image.shape
    height, width = h + 2, w + 2  # h + 3 - 1, w + 3 - 1

    edged_fft = np.multiply(fft(image, width, height), fft(kernel, width, height))
    edged = ifft(edged_fft)
    drawer(edged, 'edged_fft')

    os.chdir('../..')
