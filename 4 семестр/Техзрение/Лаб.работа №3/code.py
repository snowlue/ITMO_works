import os
import cv2
from cv2.typing import MatLike
import numpy as np
import scipy.signal
import skimage
from functools import partial

os.makedirs('images', exist_ok=True)  # Создаём папку images, если её нет
os.chdir('images')  # Переходим в папку images, чтобы сохранять результаты в неё
np.seterr(invalid='ignore', divide='ignore')  # Игнорируем лишние предупреждения


def additive_noise(image: MatLike):
    """Аддитивный шум"""
    exp_dist = np.random.default_rng().exponential(5, image.shape)
    return (image.astype(np.float32) + exp_dist).clip(0, 255).astype(np.uint8) / 255


def multiplicative_noise(image: MatLike):
    """Мультипликативный шум"""
    uniform_dist = np.random.default_rng().uniform(0, 1, image.shape)
    return (image.astype(np.float32) * uniform_dist).clip(0, 255).astype(np.uint8) / 255


def apply_noise(img: MatLike, name: str):
    """Применяет к изображению выбранный в name шум"""
    method_func = {
        'impulse': partial(skimage.util.random_noise, mode='s&p', amount=0.01, salt_vs_pepper=0.75),
        'additive': additive_noise,
        'multiplicative': multiplicative_noise,
        'speckle': partial(skimage.util.random_noise, mode='speckle', var=0.1, mean=0.1),
        'gaussian': partial(skimage.util.random_noise, mode='gaussian', var=0.09, mean=0.2),
        'poisson': partial(skimage.util.random_noise, mode='poisson'),
    }[name]
    cv2.imwrite(f'{name}.jpg', method_func(img) * 255)


def contraharmonic_filter(image: MatLike, Q: int):
    """Контргармоничный фильтр"""
    kernel = np.ones((3, 3), dtype=np.float32)
    num, den = np.power(image, Q + 1), np.power(image, Q)
    filtered_image = np.divide(
        cv2.filter2D(src=num, ddepth=-1, kernel=kernel), cv2.filter2D(src=den, ddepth=-1, kernel=kernel)
    )
    return filtered_image.clip(0, 255).astype(np.uint8)


def rang_filter(image: MatLike, k: int, rank: int):
    """Ранговый фильтр"""
    kernel = np.ones((k, k), dtype=np.float32)
    rows, cols = image.shape[0:2]

    bordered_image = image.astype(np.float32) / 255 if image.dtype == np.uint8 else image
    bordered_image = cv2.copyMakeBorder(
        bordered_image,
        int((k - 1) / 2),
        int(k / 2),
        int((k - 1) / 2),
        int(k / 2),
        cv2.BORDER_REPLICATE,
    )

    I_layers = np.zeros(image.shape + (k**2,), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            I_layers[:, :, i * k + j] = kernel[i, j] * bordered_image[i : i + rows, j : j + cols]

    I_layers.sort()

    filtered_image = I_layers[:, :, rank]

    return (255 * filtered_image).clip(0, 255).astype(np.uint8)


def wiener_filter(image: MatLike, k: int):
    """Винеровский фильтр"""
    rows, cols = image.shape[0:2]
    kernel = np.ones((k, k))

    img_copy = image.astype(np.float32) / 255 if image.dtype == np.uint8 else image
    img_copy = cv2.copyMakeBorder(
        img_copy,
        int((k - 1) / 2),
        int(k / 2),
        int((k - 1) / 2),
        int(k / 2),
        cv2.BORDER_REPLICATE,
    )

    bgr_planes = cv2.split(img_copy)
    bgr_planes_2 = []
    k_power = np.power(kernel, 2)
    for plane in bgr_planes:
        plane_power = np.power(plane, 2)
        m, q = np.zeros(image.shape[0:2], np.float32), np.zeros(image.shape[0:2], np.float32)
        for i in range(k):
            for j in range(k):
                m = m + kernel[i, j] * plane[i : i + rows, j : j + cols]
                q = q + k_power[i, j] * plane_power[i : i + rows, j : j + cols]
        m = m / np.sum(kernel)
        q = q - m**2
        v = np.sum(q) / image.size

        plane_2 = plane[(k - 1) // 2 : (k - 1) // 2 + rows, (k - 1) // 2 : (k - 1) // 2 + cols]
        plane_2 = np.where(q < v, m, (plane_2 - m) * (1 - v / q) + m)
        bgr_planes_2.append(plane_2)

    filtered_image = cv2.merge(bgr_planes_2)
    filtered_image = (255 * filtered_image).clip(0, 255).astype(np.uint8) if image.dtype == np.uint8 else filtered_image

    return filtered_image


def calculate_intensity(img_padded_float, i, j, s, s_max):
    """Подсчитывает интенсивность пикселя для адаптивного медианного фильтра"""
    # Extract window
    window = img_padded_float[i - (s // 2) : i + (s // 2) + 1, j - (s // 2) : j + (s // 2) + 1]
    # Calculate necessary values
    z_min = np.min(window)
    z_max = np.max(window)
    z_med = np.sort(window.reshape(-1))[window.size // 2]

    # Check condition 1
    if z_min < z_med < z_max:
        h, w = window.shape
        z = window[h // 2, w // 2]
        # Check condition 2
        if z_min < z < z_max:
            return z
        else:
            return z_med
    else:
        # Increase size of the window
        s += 2
        if s <= s_max:
            return calculate_intensity(img_padded_float, i, j, s, s_max)
        else:
            return img_padded_float[i, j]


def adaptive_median_filter(img, s_start, s_max):
    """Адаптивный медианный фильтр"""
    n_rows, n_cols = img.shape[:2]
    img_padded_float = img.astype(np.float32) / 255
    offset = s_max // 2
    img_padded_float = cv2.copyMakeBorder(img_padded_float, offset, offset, offset, offset, cv2.BORDER_REPLICATE)
    filtered_img_padded = np.zeros_like(img_padded_float, dtype=np.float32)

    # Go through all pixels
    for i in range(offset, n_rows + offset + 1):
        for j in range(offset, n_cols + offset + 1):
            filtered_img_padded[i, j] = calculate_intensity(img_padded_float, i, j, s_start, s_max)

    filtered_img = filtered_img_padded[offset:-offset, offset:-offset]

    # Convert back
    return np.clip(filtered_img * 255, 0, 255).astype(np.uint8)


class Parameters(tuple):
    """Принимает на вход именнованные аргументы с итерируемыми объектами.\n
    При обращении по индексу возвращает словарь с ключами и значениями,
    которые находятся в итерируемых объектах по этому индексу.\n
    ### Пример:
    ```py
    >>> params = Parameters(a=[2, 4, 6, 8], b=[1, 3, 5, 7], c=[3, 6])
    >>> params[0]
    {'a': 2, 'b': 1, 'c': 3}
    >>> params[2]
    {'a': 6, 'b': 5, 'c': 6}
    >>> params[100]
    {'a': 8, 'b': 7, 'c': 6}
    ```"""

    def __init__(self, **kwargs):
        self._objects = kwargs

    def __getitem__(self, index):
        return {key: value[min(index, len(value) - 1)] for key, value in self._objects.items()}


def apply_filter(img: MatLike, name: str, set_num: int, prefix: str):
    """Применяет к изображению выбранный в name низкочастотный или нелинейный фильтр
    с набором коэффициентов под номером set_num"""
    params = {
        'gaussian': Parameters(ksize=((3, 3),), sigmaX=(0,)),
        'contrharmonic': Parameters(Q=(-1.5, -0.5, 0.5, 1.5)),
        'median': Parameters(ksize=(3, 5, 9)),
        '2d-median': Parameters(kernel_size=(3, 5, 9)),
        'rang': Parameters(k=(3, 3, 3, 5), rank=(1, 4, 7, 1, 11, 21)),
        'wiener': Parameters(k=(3, 5, 9)),
        'adaptive median': Parameters(s_start=(3, 5), s_max=(7, 17)),
    }[name][set_num]
    method_func = {
        'gaussian': cv2.GaussianBlur,
        'contrharmonic': contraharmonic_filter,
        'median': cv2.medianBlur,
        '2d-median': scipy.signal.medfilt2d,
        'rang': rang_filter,
        'wiener': wiener_filter,
        'adaptive median': adaptive_median_filter,
    }[name]
    params_string = ', '.join([f'{key}={value}' for key, value in params.items()])
    cv2.imwrite(f'{prefix} - {name} ({params_string}).jpg', method_func(img, **params))


def robertson_detection(src_image):
    g_x, g_y = np.array([[1, -1], [0, 0]]), np.array([[1, 0], [-1, 0]])
    i_x, i_y = cv2.filter2D(src_image, -1, g_x), cv2.filter2D(src_image, -1, g_y)

    abs_grad_x, abs_grad_y = cv2.convertScaleAbs(i_x), cv2.convertScaleAbs(i_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def prewitt_detection(src_image):
    g_x, g_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    i_x, i_y = cv2.filter2D(src_image, -1, g_x), cv2.filter2D(src_image, -1, g_y)

    abs_grad_x, abs_grad_y = cv2.convertScaleAbs(i_x), cv2.convertScaleAbs(i_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def sobel_detection(src_image):
    i_x, i_y = cv2.Sobel(src_image, cv2.CV_16S, 1, 0), cv2.Sobel(src_image, cv2.CV_16S, 0, 1)

    abs_grad_x, abs_grad_y = cv2.convertScaleAbs(i_x), cv2.convertScaleAbs(i_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def laplassian_detection(src_image):
    g_d = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return cv2.filter2D(src_image, -1, g_d)


def apply_detection(img: MatLike, name: str):
    """Применяет к изображению выбранный в name высокочастотный фильтр выделения контуров"""
    method_func = {
        'robertson': robertson_detection,
        'prewitt': prewitt_detection,
        'sobel': sobel_detection,
        'laplassian': laplassian_detection,
        'canny': lambda image: cv2.Canny(image, 100, 200),
    }[name]
    cv2.imwrite(f'{name}.jpg', method_func(img))


image = cv2.imread('photo.jpg', 0)

# Создаём папки для сохранения результата
for dirname in ('1_noise', '2_low_filters', '3_nonlinear_filters', '4_high_filters'):
    os.makedirs(dirname, exist_ok=True)


# <---- 1. шумы ---->
os.chdir('1_noise')
apply_noise(image, 'impulse')
apply_noise(image, 'additive')
apply_noise(image, 'multiplicative')
apply_noise(image, 'speckle')
apply_noise(image, 'gaussian')
apply_noise(image, 'poisson')

# # <---- 2. низкочастотная фильтрация ---->
os.chdir('../2_low_filters')
for noise_type in ('additive', 'gaussian', 'impulse', 'multiplicative', 'poisson', 'speckle'):
    noisy_image = cv2.imread(filename=f'../1_noise/{noise_type}.jpg')
    for set_num in range(4):
        apply_filter(noisy_image, 'gaussian', set_num, noise_type)
        apply_filter(noisy_image, 'contrharmonic', set_num, noise_type)


# # <---- 3. нелинейная фильтрация ---->
os.chdir('../3_nonlinear_filters')
for noise_type in ('additive', 'gaussian', 'impulse', 'multiplicative', 'poisson', 'speckle'):
    noisy_image = cv2.imread(f'../1_noise/{noise_type}.jpg', cv2.IMREAD_GRAYSCALE)
    for set_num in range(6):
        apply_filter(noisy_image, 'median', set_num, noise_type)
        apply_filter(noisy_image, '2d-median', set_num, noise_type)
        apply_filter(noisy_image, 'rang', set_num, noise_type)
        apply_filter(noisy_image, 'wiener', set_num, noise_type)
        apply_filter(noisy_image, 'adaptive median', set_num, noise_type)


# # <---- 4. высокочастотная фильтрация ---->
os.chdir('../4_high_filters')
apply_detection(image, 'robertson')
apply_detection(image, 'prewitt')
apply_detection(image, 'sobel')
apply_detection(image, 'laplassian')
apply_detection(image, 'canny')
