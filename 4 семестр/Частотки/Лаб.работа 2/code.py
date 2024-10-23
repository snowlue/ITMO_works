import warnings
from typing import Callable

import librosa
import matplotlib.pyplot as plt
import numpy as np

#! Перейдите в класс Config (line:78), чтобы задать все вводные
warnings.filterwarnings('ignore', category=np.ComplexWarning)


class LabFunctions:
    """Класс, содержащий объекты, требуемые для выполнения заданий в лабораторной работе."""

    coefficients = ((1, 1), (2, 1), (2, 2))
    shift_coefficients = (-5, -2, 3, 4)
    a, b = coefficients[0]  # видоизменение исходной функции
    a, b, c = 1, 2, shift_coefficients[0]  # видоизменение исходной функции со сдвигом

    rectangular = (
        lambda t: LabFunctions.a if abs(t) <= LabFunctions.b else 0,
        list(map(lambda i: i * 5, range(-3, 4))),
        list(map(lambda i: i / 10 * 5, range(-2, 8))),
    )
    triangular = (
        lambda t: (LabFunctions.a - abs(LabFunctions.a * t / LabFunctions.b)) if abs(t) <= LabFunctions.b else 0,
        list(map(lambda i: i * 5, range(-3, 4))),
        list(map(lambda i: i / 10 * 2, range(-2, 10))),
    )
    csin = (
        lambda t: LabFunctions.a * np.sinc(LabFunctions.b * t),
        list(map(lambda i: i * 2, range(-4, 5))),
        list(map(lambda i: i / 10 * 2, range(-1, 6))),
    )
    gaussian = (
        lambda t: LabFunctions.a * np.exp(-LabFunctions.b * t**2),
        list(map(lambda i: i * 2, range(-5, 6))),
        list(map(lambda i: i / 4, range(7))),
    )
    fade = (
        lambda t: LabFunctions.a * np.exp(-LabFunctions.b * abs(t)),
        list(map(lambda i: i * 5, range(-5, 6))),
        list(map(lambda i: i / 5, range(10))),
    )

    triangular_shift = (
        lambda t: (LabFunctions.a - abs(LabFunctions.a * (t + LabFunctions.c) / LabFunctions.b))
        if abs(t + LabFunctions.c) <= LabFunctions.b
        else 0,
        list(map(lambda i: i * 2, range(-5, 6))),
        list(map(lambda i: i / 10 * 2, range(-5, 6))),
    )

    accord_file = (
        'sources/7_accord/Аккорд (2).mp3',
        list(map(lambda i: i * 100, [0, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18])),
        list(map(lambda i: i / 1000, range(0, 7))),
    )


class Config:
    """Конфиг, содержащий все вводные для программы."""

    FUNC_TYPE = 'accord'  # ← Задайте здесь тип функции ↓
    #                                                'real'    для вещественной функции
    #                                                'complex' для комплексной функции
    #                                                'accord'  для файла со звуком аккорда

    if FUNC_TYPE == 'accord':
        file_name, fxticks, fyticks = LabFunctions.accord_file
        _amplitudes, _source_rate = librosa.load(file_name)
        source_function = np.vectorize(lambda t: Config._amplitudes[int(t * Config._source_rate)])
    else:
        source_function, fxticks, fyticks = LabFunctions.triangular_shift  # ← Задайте здесь функцию

    TASK = 'abs'  # ← Задайте здесь задание ↓
    #                                      'graph'    для вывода графика функции
    #                                      'fourier'  для получения графика Фурье-образа
    #                                      'abs'      для получения графика модуля Фурье-образа
    #                                      'parseval' для проверки Парсеваля

    # ↓ Задайте здесь пределы интегрирования для проверки Парсеваля
    LIMITS = (-LabFunctions.b, LabFunctions.b)

    if FUNC_TYPE == 'complex':
        # ↓ Задайте название выходного файла (если None, то название будет совпадать с TASK)
        name = TASK + f'_{LabFunctions.shift_coefficients.index(LabFunctions.c) + 1}'
    elif FUNC_TYPE == 'accord':
        name = TASK
    else:
        # ↓ Задайте название выходного файла (если None, то название будет совпадать с TASK)
        name = TASK + f'_{LabFunctions.coefficients.index((LabFunctions.a, LabFunctions.b)) + 1}'

    xticks = list(map(lambda i: i / 10 * 2, range(11)))  # ← Задайте здесь значения x, которые нужно отметить на графике
    yticks = list(map(lambda i: i / 10, range(-5, 6)))  # ← Задайте здесь значения y, которые нужно отметить на графике

    if TASK in {'fourier', 'abs'}:  # ↓ Оптимальный масштаб для Фурье-образа ↓
        xticks, yticks = fxticks, fyticks
    xlabels = None  # ← Задайте здесь подписи к значениям xticks (если None, то подписи будут совпадать с xticks)
    ylabels = None  # ← Задайте здесь подписи к значениям yticks (если None, то подписи будут совпадать с yticks)


def f(x, get_original: bool = False):
    """
    Исходная функция, для которой вычисляется преобразование Фурье.

    :param x: Входные данные для функции.
    :type x: numpy.ndarray
    :param get_original: Флаг, указывающий, нужно ли получить оригинальную функцию.
    :type get_original: bool, optional
    :return: Результат вычисления функции.
    :rtype: numpy.ndarray
    """
    return (Config.source_function if get_original else np.vectorize(Config.source_function, otypes=[np.complex_]))(x)


def raw_image(start: float = min(Config.xticks), end: float = max(Config.xticks)):
    """
    Сырой Фурье-образ функции.

    :param x: Входные данные для функции.
    :type x: numpy.ndarray
    :return: Результат вычисления функции.
    :rtype: numpy.ndarray
    """
    image = lambda w: 1 / (np.sqrt(2 * np.pi)) * dot_product(f, lambda t: np.e ** (-1j * w * t), start, end)
    return np.vectorize(image)


def w_fourier_image(x, start: float = min(Config.xticks), end: float = max(Config.xticks)):
    """
    Вычисляет значение Фурье-образа (с угловой частотой w) функции в точке x.

    :param x: Входные данные для функции.
    :type x: numpy.ndarray
    :return: Результат вычисления функции.
    :rtype: numpy.ndarray
    """
    return raw_image(start, end)(x)


def v_fourier_image(x, start: float = min(Config.xticks), end: float = max(Config.xticks)):
    """
    Вычисляет значение Фурье-образа (с обыкновенной частотой v) функции в точке x.

    :param x: Входные данные для функции.
    :type x: numpy.ndarray
    :return: Результат вычисления функции.
    :rtype: numpy.ndarray
    """
    image = lambda v: dot_product(lambda t: f(t, True), lambda t: np.e ** (-2j * np.pi * v * t), start, end)
    return np.vectorize(image)(x)


def dot_product(f: Callable, g: Callable, a: float, b: float) -> np.ndarray:
    """
    Вычисляет скалярное произведение функций f и g на отрезке [a, b].

    :param f: Функция f(x).
    :type f: Callable
    :param g: Функция g(x).
    :type g: Callable
    :param a: Левая граница отрезка.
    :type a: float
    :param b: Правая граница отрезка.
    :type b: float
    :return: Скалярное произведение функций f и g на отрезке [a, b].
    :rtype: np.ndarray
    """
    x = np.linspace(a, b, 2000)  # Генерируем точки на отрезке [a, b]
    dx = x[1] - x[0]  # Шаг интегрирования
    return np.dot(f(x), g(x)) * dx  # Возвращаем скалярное произведение


def parseval_check(a: float, b: float) -> float:
    """
    Проверяет равенство Парсеваля для функции и Фурье-образа.

    :param a: Левая граница отрезка.
    :type a: float
    :param b: Правая граница отрезка.
    :type b: float
    :return: Отклонение второй нормы функции от второй нормы Фурье-образа на отрезке (a, b).
    :rtype: float
    """
    fimage = raw_image(a, b)
    abs_fimage = np.vectorize(lambda x: abs(fimage(x)))

    func_norm = abs(dot_product(f, f, a, b))
    fimage_norm = dot_product(abs_fimage, abs_fimage, -100, 100)  # по сути берём 200 коэффициентов Фурье

    return abs(func_norm - fimage_norm)


def main():
    if Config.TASK == 'parseval':
        return print('Parseval deviation:\n| ||f||^2 - ||F||^2 | = {:.7f}'.format(parseval_check(*Config.LIMITS)))

    x_values = np.linspace(min(Config.xticks), max(Config.xticks), 1000)  # Устанавливаем значения x для графика
    if Config.TASK in {'fourier', 'abs'}:
        # Подсчитываем значения y графика преобразования Фурье
        if Config.FUNC_TYPE == 'accord':
            image = v_fourier_image(x_values, 0, 0.1)
        else:
            image = w_fourier_image(x_values)
        print('Fourier image calculated!')

    # Настраиваем график, чтобы всё было по красоте
    fig, ax = plt.subplots()
    fig.set_size_inches(640 / fig.dpi, 358 / fig.dpi)
    if Config.FUNC_TYPE == 'accord':
        ax.set_xlabel({'graph': '$t$', 'fourier': '$\\nu$', 'abs': '$\\nu$'}[Config.TASK])
    else:
        ax.set_xlabel({'graph': '$t$', 'fourier': '$\\omega$', 'abs': '$\\omega$'}[Config.TASK])
    if Config.FUNC_TYPE == 'complex':
        ax.set_ylabel(
            {'graph': '$g(t)$', 'fourier': '$\\hat{g}(\\omega)$', 'abs': '$|\\hat{g}(\\omega)|$'}[Config.TASK]
        )
    elif Config.FUNC_TYPE == 'accord':
        ax.set_ylabel({'graph': '$g(t)$', 'fourier': '$\\hat{f}(\\nu)$', 'abs': '$|\\hat{f}(\\nu)|$'}[Config.TASK])
    else:
        ax.set_ylabel({'graph': '$f(t)$', 'fourier': '$\\hat{f}(\\omega)$', 'abs': ''}[Config.TASK])
    ax.set_xlim(min(0, min(Config.xticks)), max(0, max(Config.xticks)))
    ax.set_ylim(min(0, min(Config.yticks)), max(0, max(Config.yticks)))
    ax.axvline(0, color='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(Config.xticks)
    ax.set_xticklabels(Config.xlabels or Config.xticks)
    ax.set_yticks(Config.yticks)
    ax.set_yticklabels(Config.ylabels or Config.yticks)
    ax.grid()

    # Отрисовываем графики
    if Config.TASK == 'abs':
        if Config.FUNC_TYPE == 'complex':
            ax.plot(x_values, np.abs(image), color='#883aff', label='$|\\hat{g}(\\omega)|$')
        elif Config.FUNC_TYPE == 'accord':
            ax.plot(x_values, np.abs(image), color='#883aff', label='$|\\hat{f}(\\nu)|$')
        else:
            ax.plot(x_values, np.abs(image), color='#883aff', label='$|\\hat{f}(\\omega)|$')
    elif Config.TASK == 'fourier':
        if Config.FUNC_TYPE == 'complex':
            ax.plot(x_values, image.real, color='#34D1BF', label='Re $\\hat{g}(\\omega)$')
            ax.plot(x_values, image.imag, color='#D1345B', label='Im $\\hat{g}(\\omega)$')
        elif Config.FUNC_TYPE == 'accord':
            ax.plot(x_values, image, color='#34D1BF', label='$\\hat{f}(\\nu)$')
        else:
            ax.plot(x_values, image, color='#34D1BF', label='$\\hat{f}(\\omega)$')
    else:
        if Config.FUNC_TYPE == 'complex':
            ax.plot(x_values, f(x_values), color='#3454D1', label='$g(t)$')
        elif Config.FUNC_TYPE == 'accord':
            ax.plot(x_values, f(x_values, True), color='#3454D1', label='$f(t)$')
        else:
            ax.plot(x_values, f(x_values), color='#3454D1', label='$f(t)$')

    # Устанавливаем положение легенды, тонкие границы вокруг графика, включаем Tex и сохраняем график
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.rcParams.update({'text.usetex': True})
    plt.savefig(f'{Config.name or Config.TASK}.png')
    print('Graph saved!')


if __name__ == '__main__':
    main()
