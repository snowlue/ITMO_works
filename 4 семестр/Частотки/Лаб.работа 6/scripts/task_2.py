import os

import cv2 as cv
import numpy as np

from .drawer import drawer


def fft(image, width, height):
    return np.fft.fft2(image, (height, width))


def ifft(image):
    return np.fft.ifft2(image).real


def blurring(image, option, n) -> np.ndarray:  # type: ignore
    match option:
        case 'block':
            kernel = np.ones((n, n), np.float64) / n**2
            return cv.filter2D(image, ddepth=-1, kernel=kernel)
        case 'gauss':
            gauss = np.array(
                [
                    [
                        np.e ** ((-9 / (n**2)) * ((i - (n + 1) / 2) ** 2 + (j - (n + 1) / 2) ** 2))
                        for i in range(1, n + 1)
                    ]
                    for j in range(1, n + 1)
                ]
            )
            kernel = gauss / np.sum(gauss)
            return cv.filter2D(image, ddepth=-1, kernel=kernel)


def task(filename: str):
    os.makedirs('sources/2second', exist_ok=True)
    os.chdir('sources/2second')

    ns = [9, 15, 27]
    image = cv.imread(f'../{filename}', cv.IMREAD_GRAYSCALE)
    h, w = image.shape
    for n in ns:
        blurred_block = blurring(image, 'block', n)
        blurred_gauss = blurring(image, 'gauss', n)
        drawer(blurred_block, f'block_{n}')
        drawer(blurred_gauss, f'gauss_{n}')

    for n in ns:
        height, width = h + n - 1, w + n - 1
        gauss = np.array(
            [
                [np.e ** ((-9 / (n**2)) * ((i - (n + 1) / 2) ** 2 + (j - (n + 1) / 2) ** 2)) for i in range(1, n + 1)]
                for j in range(1, n + 1)
            ]
        )

        block_kernel = np.ones((n, n), np.float64) / n**2
        fft_block = np.multiply(fft(image, width, height), fft(block_kernel, width, height))
        blurred_block = ifft(fft_block)

        gauss_kernel = gauss / np.sum(gauss)
        fft_gauss = np.multiply(fft(image, width, height), fft(gauss_kernel, width, height))
        blurred_gauss = ifft(fft_gauss)

        drawer(blurred_block, f'block_fft_{n}')
        drawer(blurred_gauss, f'gauss_fft_{n}')

    os.chdir('../..')
