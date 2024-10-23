import os

import matplotlib.image as img
import numpy as np

from .drawer import drawer


def fft(image):
    image_copy = np.copy(image)
    image_r = np.fft.fftshift(np.fft.fft2(image_copy[:, :, 0]))
    image_g = np.fft.fftshift(np.fft.fft2(image_copy[:, :, 1]))
    image_b = np.fft.fftshift(np.fft.fft2(image_copy[:, :, 2]))
    fourier_image = np.stack([image_r, image_g, image_b], axis=2)
    return fourier_image


def ifft(image, angle, log_max):
    fourier_image = np.exp(1j * angle) * (np.exp(image * log_max) - 1)

    photo_r = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 0]))
    photo_g = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 1]))
    photo_b = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 2]))
    restored_photo = np.stack([photo_r, photo_g, photo_b], axis=2)
    restored_photo = np.clip(restored_photo, 0, 1)
    return restored_photo.real


def task(filename: str):
    os.makedirs('sources/1first', exist_ok=True)
    os.chdir('sources/1first')

    image = img.imread(f'../{filename}')
    fft_image = fft(image)

    fft_abs = np.abs(fft_image)
    fft_log = np.log(fft_abs + 1)
    fft_log_max = np.max(fft_log)
    fft_log /= fft_log_max

    drawer(fft_log, '11_fft')

    # Восстановление изображения после редактирования образа
    if os.path.exists('11_fft_edited.png'):
        edited = img.imread('11_fft_edited.png')[:, :, :3]
        filtered = ifft(edited, np.angle(fft_image), fft_log_max)
        drawer(filtered, '11_filtered')

    os.chdir('../..')
