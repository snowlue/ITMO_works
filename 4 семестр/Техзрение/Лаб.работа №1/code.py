import os
from typing import Callable, Sequence
import cv2
from cv2.typing import MatLike
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray

os.makedirs('images', exist_ok=True)  # Создаём папку images, если её нет
os.chdir('images')  # Переходим в папку images, чтобы сохранять результаты в неё
np.seterr(invalid='ignore')  # Игнорируем предупреждения, появляющиеся при вычислении логарифмов через np.log
plt.figure(figsize=(8, 6))


def clear():
    """Очищает холст matplotlib от графиков"""
    plt.cla()  # Очищаем график
    plt.clf()  # Очищаем фигуру
    plt.subplots_adjust(left=0.07, right=0.95)


def calc_rgb_hist(channels: Sequence[MatLike]) -> list[MatLike]:
    """Вычисляет нормализованные гистограммы изображения"""
    hists = []
    for i in range(3):
        hist = cv2.calcHist(channels, [i], None, [hist_size], hist_range)
        hists.append(hist / (image.shape[0] * image.shape[1]))  # Нормализуем гистограмму
    return hists


def calc_cumulative_hist(channels: Sequence[MatLike]) -> tuple[NDArray, ...]:
    """Вычисляет кумулятивную гистограмму изображения"""
    # Подсчитываем нормализованные гистограммы по каналам RGB, применяем к ним np.cumsum и возвращаем в кортеже
    return tuple(map(np.cumsum, calc_rgb_hist(channels)))


def render_rgb_hist(channels: Sequence[MatLike], name: str):
    """Рендерит гистограмму изображения из нормализованных гистограмм каждого канала"""
    clear()
    hists = calc_rgb_hist(channels)

    plt.grid(True)
    plt.xlim(hist_range)
    plt.ylim(-0.001, 0.1)
    for hist, color in zip(hists[::-1], ('r', 'g', 'b')):
        plt.plot(hist, color=color)

    plt.title('RGB-гистограма')
    plt.savefig(f'{name}.png')


def render_cumulative_hist(channels: Sequence[MatLike], name: str):
    """Рендерит кумулятивную гистограмму изображения"""
    clear()
    cumhists = calc_cumulative_hist(channels)

    plt.grid(True)
    plt.xlim(hist_range)
    plt.ylim(-0.005, 1.005)
    for hist, color in zip(cumhists[::-1], ('r', 'g', 'b')):
        plt.plot(hist, color=color)

    plt.title('Кумулятивная гистограмма')
    plt.savefig(f'{name}.png')


def apply_method(method: Callable, image: MatLike, output_dir: str, **kwargs):
    """Применяет преобразование к изображению, компрессирует значения выше
    максимальной яркости до 255 и сохраняет результат в папку output_dir"""
    new_image = method(image, **kwargs)
    cv2.threshold(new_image, 255, 255, cv2.THRESH_TRUNC, new_image)
    new_image = new_image.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/photo.jpg', new_image)

    # Отрендерим гистограммы преобразованного изображения
    render_rgb_hist(cv2.split(new_image), f'{output_dir}/hist')
    render_cumulative_hist(cv2.split(new_image), f'{output_dir}/cumhist')


def linear_alignment(image: MatLike) -> NDArray[np.uint8]:
    """Линейное выравнивание гистограммы"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []  # Будем формировать новые каналы в этот список
    for channel, cumhist in zip(channels, cumhists):
        # Применяем линейное выравнивание к каждому каналу по кумулятивной гистограмме
        # В гистограмме диапазон [0..1], поэтому умножаем на 255
        new_channels.append(255 * cumhist[channel])
    return cv2.merge(new_channels)


def arithmetic_operations(image: MatLike, shifts: list[float]) -> NDArray[np.uint8]:
    """Арифметические операции с изображением"""
    channels = list(cv2.split(image))
    for i, channel in enumerate(channels):
        # Сдвигаем каждый канал на значение shift
        channels[i] = np.clip(channel.astype(np.uint16) + shifts[i], 0, 255).astype(np.uint8)
    return cv2.merge(channels)


def contrast_stretching(image: MatLike, alpha: float) -> NDArray[np.uint8]:
    """Расширение контраста методом растяжения динамического диапазона"""
    # Если изображение в целочисленном диапазоне [0..255], то конвертируем в вещественный [0..1]
    image = image.astype(np.float64) / 255 if image.dtype == 'uint8' else image
    channels = list(cv2.split(image))
    for i, channel in enumerate(channels):
        # Применяем растяжение динамического диапазона к каждому каналу
        cmin, cmax = channel.min(), channel.max()
        channels[i] = ((channel - cmin) / (cmax - cmin)) ** alpha * 255
    image = cv2.merge(channels)
    # Если изображение было в целочисленном диапазоне, то возвращаем его обратно в этот диапазон
    return (image.clip(0, 255) if image.dtype == 'uint8' else image).astype(np.uint8)


def uniform_conversion(image: MatLike) -> NDArray[np.uint8]:
    """Равномерное преобразование гистограммы"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []
    for channel, cumhist in zip(channels, cumhists):
        cmin, cmax = channel.min(), channel.max()
        new_channels.append((cmax - cmin) * cumhist[channel] + cmin)
    return cv2.merge(new_channels)


def exponential_conversion(image: MatLike, alpha: float) -> NDArray[np.uint8]:
    """Экспоненциальное преобразование гистограммы"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []
    for channel, cumhist in zip(channels, cumhists):
        cmin = channel.min()
        new_channels.append(cmin - 255 / alpha * np.log(1 - cumhist[channel]))
    return cv2.merge(new_channels)


def rayleigh_conversion(image: MatLike, alpha: float) -> NDArray[np.uint8]:
    """Преобразование гистограммы по закону Рэлея"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []
    for channel, cumhist in zip(channels, cumhists):
        cmin = channel.min()
        new_channels.append(cmin + np.sqrt(2 * alpha**2 * np.log(1 / (1 - cumhist[channel]))) * 255)
    return cv2.merge(new_channels)


def rule23_conversion(image: MatLike) -> NDArray[np.uint8]:
    """Преобразование гистограммы по закону степени 2/3"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []
    for channel, cumhist in zip(channels, cumhists):
        new_channels.append(cumhist[channel] ** (2 / 3) * 255)
    return cv2.merge(new_channels)


def hyperbolic_conversion(image: MatLike, alpha: float) -> NDArray[np.uint8]:
    """Преобразование гистограммы по закону степени 2/3"""
    channels = cv2.split(image)
    cumhists = calc_cumulative_hist(channels)
    new_channels = []
    for channel, cumhist in zip(channels, cumhists):
        new_channels.append(alpha ** cumhist[channel] * 255)
    return cv2.merge(new_channels)


def LUT_conversion(image: MatLike, alpha: float) -> NDArray[np.uint8]:
    """Преобразование изображения с помощью таблицы преобразования"""
    lut = np.arange(256, dtype=np.uint8)
    lut = (lut - image.min()) / (image.max() - image.min())
    lut = np.where(lut > 0, lut, 0)
    lut = np.clip(255 * np.power(lut, alpha), 0, 255)
    return cv2.LUT(image, lut)


def render_image_profile(img: MatLike, name: str):
    """Рендерит профиль яркости изображения"""
    clear()
    profile = img[img.shape[0] // 2, :]

    fig = plt.figure()
    fig.set_size_inches(654 / fig.dpi, 744 / fig.dpi)

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.set_title('Исходное изображение')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')

    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
    ax2.set_xlabel('Профиль яркости', fontsize='large')
    ax2.plot(profile)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.savefig(f'{name}_profile.png')


def render_projection(img: MatLike, name: str):
    """Рендерит проекцию изображения"""
    clear()
    channels = cv2.split(img)
    projection_x = sum(np.sum(channel, axis=0) for channel in channels) / img.shape[0] / 3
    projection_y = sum(np.sum(channel, axis=1) for channel in channels) / img.shape[1] / 3

    fig = plt.figure()
    fig.set_size_inches(600 / fig.dpi, 840 / fig.dpi)

    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    ax1.set_title('Исходное изображение')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')

    ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2, sharex=ax1)
    ax2.set_xlabel('Проекция на X', fontsize='large')
    ax2.plot(range(img.shape[1]), projection_x)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=ax1)
    ax3.set_title('Проекция на Y')
    ax3.plot(projection_y, range(img.shape[0]))
    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.savefig(f'{name}_projection.png')


hist_size = 256
hist_range = (0, hist_size)

image = cv2.imread('photo.jpg')
BGR = cv2.split(image)

# Посмотрим на гистограммы изображения
render_rgb_hist(BGR, 'hist')
render_cumulative_hist(BGR, 'cumhist')


# Создаём папки для сохранения результата
for dirname in (
    '1aligned',
    '2shift',
    '3stretched',
    '4uniform',
    '5exponential',
    '6rayleigh',
    '7rule23',
    '8hyperbolic',
    '9LUT',
):
    os.makedirs(dirname, exist_ok=True)


# <---- 1. линейное выравнивание гистограммы ---->
# Линейно выравниваем гистограмму изображения
apply_method(linear_alignment, image, '1aligned')

# <---- 2. арифметические операции ---->
# Сдвигаем каждый канал изображения на заданное количество градаций вправо
apply_method(arithmetic_operations, image, '2shift', shifts=[65, 65, 50])

# <---- 3. расширение контраста ---->
# Применяем растяжение динамического диапазона с коэффициентом α = 0.8
apply_method(contrast_stretching, image, '3stretched', alpha=0.8)

# <---- 4. равномерное преобразование ---->
# Применяем равномерное преобразование гистограммы
apply_method(uniform_conversion, image, '4uniform')

# <---- 5. экспоненциальное преобразование ---->
# Применяем экспоненциальное преобразование гистограммы с коэффициентом α = 4
apply_method(exponential_conversion, image, '5exponential', alpha=4)

# <---- 6. преобразование по закону Рэлея ---->
# Применяем рэлеевское преобразование гистограммы с коэффициентом α = 0.4
apply_method(rayleigh_conversion, image, '6rayleigh', alpha=0.4)

# <---- 7. преобразование по закону степени 2/3 ---->
# Применяем преобразование гистограммы по закону степени 2/3
apply_method(rule23_conversion, image, '7rule23')

# <---- 8. гиперболическое преобразование ---->
# Применяем гиперболическое преобразование гистограммы с коэффициентом α = 0.2
apply_method(hyperbolic_conversion, image, '8hyperbolic', alpha=0.04)

# <---- 9. таблица поиска ---->
# Применяем таблицу поиска к изображению с коэффициентом α = 0.5
apply_method(LUT_conversion, image, '9LUT', alpha=0.5)

# <---- 10. профиль яркости ---->
# Посмотрим на профиль яркости изображения
barcode_image = cv2.imread('barcode.png')
render_image_profile(barcode_image, 'barcode')

# <---- 11. проекции ---->
# Посмотрим на проекции изображения на оси X и Y
passport_image = cv2.imread('passport.jpg')
render_projection(passport_image, 'passport')
