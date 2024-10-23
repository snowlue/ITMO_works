import warnings
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray

#! Перейдите в класс Config (line:78), чтобы задать все вводные
warnings.filterwarnings('ignore', category=np.ComplexWarning)

ц
def create_parametric_func(R: float, T: float) -> Callable[[float], complex]:
    """
    `[Изменяемая]`\n
    Создаёт параметрическую функцию, которая возвращает комплексное число в зависимости от параметра t.

    :param R: Константа R, требуемая по заданию.
    :type R: float
    :param T: Константа T, требуемая по заданию.
    :type T: float
    :return: Функция, которая принимает параметр t и возвращает комплексное число.
    :rtype: function
    """

    def pfunc_instance(t):
        t = (t + T / 8) % T - T / 8
        # Поведение для вещественной компоненты
        if -T / 8 <= t < T / 8:
            real = R
        elif T / 8 <= t < 3 * T / 8:
            real = 2 * R - 8 * R * t / T
        elif 3 * T / 8 <= t < 5 * T / 8:
            real = -R
        elif 5 * T / 8 <= t <= 7 * T / 8:
            real = -6 * R + 8 * R * t / T

        # Поведение для мнимой компоненты
        if -T / 8 <= t < T / 8:
            imag = 8 * R * t / T
        if T / 8 <= t < 3 * T / 8:
            imag = R
        if 3 * T / 8 <= t < 5 * T / 8:
            imag = 4 * R - 8 * R * t / T
        if 5 * T / 8 <= t <= 7 * T / 8:
            imag = -R

        return real + 1j * imag

    return pfunc_instance


class LabFunctions:
    """Класс, содержащий объекты, требуемые для выполнения заданий в лабораторной работе."""

    @staticmethod
    def naive_piecewise(x, pos='start'):
        """То же, что и piecewise, но на границах возвращает разные значения в зависимости
        от рассматриваемой точки куска. Необходимо для построения красивого графика кусочной функции."""
        x -= 1
        x %= 3
        match pos:
            case 'start':
                if 0 <= x < 1:
                    return 2
            case 'end':
                if 0 < x <= 1:
                    return 2
        return 3

    dots = [-10, -8, -7, -5, -4, -2, -1, 0, 1, 2, 4, 5, 7, 8, 10]
    piecewise = lambda t: 2 if 0 <= (t - 1) % 3 < 1 else 3
    even_func = lambda t: 2 * np.abs(np.sin(t))
    odd_func = lambda t: ((t - 2) % 4 - 2) ** 3
    periodic_func = lambda t: ((t / np.pi) % 2) ** 2
    parametric_func = create_parametric_func(2, 8)


class Config:
    """Конфиг, содержащий все вводные для программы."""

    source_function = LabFunctions.parametric_func  # ← Задайте здесь функцию
    N, T = 10, 8  # ← Задайте здесь порядок разложения и период функции

    GRAPH_TYPE = 'parametric'  # ← Задайте здесь тип графика ↓
    #                                                   'square_wave' для кусочной функции
    #                                                   'function'    для обычной функции
    #                                                   'parametric'  для параметрической комплексной функции
    #! Для кусочной функции можно задать поведение графика в точках разрыва в функции calculate_graph
    #! Для параметрической функции можно задать поведение в функции create_parametric_func
    #!                                   и выбрать график к построению (line:328, line:337)

    PARAMETRIC_TO_DRAW = 'real'  # ← Задайте тип графика для парам.функции ↓
    #                                                                  'complex' для [x: Re f(t), y: Im f(t)]
    #                                                                  'imag'    для [x: t, y: Im f(t)]
    #                                                                  'real'    для [x: t, y: Re f(t)]

    TASK = 'parseval'  # ← Задайте здесь задание ↓
    #                                       'empty'    для вывода графика функции
    #                                       'coeffs'   для получения коэффициентов Фурье
    #                                       'fourier'  для получения графика разложения в ряды Фурье
    #                                       'parseval' для проверки Парсеваля
    xticks = range(-10, 11)  # ← Задайте здесь значения x, которые нужно отметить на графике
    yticks = range(-4, 5)  # ← Задайте здесь значения y, которые нужно отметить на графике
    xlabels = None  # ← Задайте здесь подписи к значениям xticks (если None, то подписи будут совпадать с xticks)
    ylabels = None  # ← Задайте здесь подписи к значениям yticks (если None, то подписи будут совпадать с yticks)


def f(x, get_original: bool = False):
    """
    Функция, для которой вычисляются коэффициенты Фурье.

    :param x: Входные данные для функции.
    :type x: numpy.ndarray
    :param get_original: Флаг, указывающий, нужно ли получить оригинальную функцию.
    :type get_original: bool, optional
    :return: Результат вычисления функции.
    :rtype: numpy.ndarray
    """
    return (Config.source_function if get_original else np.vectorize(Config.source_function))(x)


def calculate_graph(x: NDArray[np.floating[Any]] | float, pos='start'):
    """
    `[Изменяемая]`\n
    Специально заданное поведение функции, по которому будет строиться её график.
    Необходимо для построения красивого графика кусочной функции

    :param x: Точка (набор точек) x функции.
    :type x: numpy.ndarray | float
    :param pos: Позиция, на которой вычисляется значение куска кусочной функции, по умолчанию `start`.
    :type pos: str, optional
    :return: Значение (набор значений) y функции в точке(-ах) x.
    :rtype: numpy.ndarray | float
    """
    # f(x, True) возвращает оригинальную функцию для построения графика обычных функций — её можно не трогать
    # LabFunctions.naive_piecewise(x, pos) возвращает кусочную функцию с разными значениями на границах

    # Вы можете сделать свою похожую здесь — для этого достаточно выставить противоположную строгость
    # неравенств в зависимости от значения pos, которое может быть только 'start' или 'end'.
    if Config.GRAPH_TYPE == 'square_wave':
        return LabFunctions.naive_piecewise(x, pos)
    return f(x, True)


def w(T: float) -> float:
    """
    Вычисляет постоянную ω для периода T.

    :param T: Период функции.
    :type T: float
    :return: Значение постоянной ω.
    :rtype: float
    """
    return 2 * np.pi / T


def parseval_check():
    """
    Проверяет равенство Парсеваля для функции с N коэффициентами Фурье. (N берётся из конфига)

    :return: Кортеж из двух значений — отклонение суммы квадратов норм коэффициентов Фурье `|a_i|^2 + |b_i|^2`
    и `|c_i|^2` от квадрата нормы функции.
    :rtype: tuple
    """
    abs_func = np.vectorize(lambda x: abs(f(x)))

    norm_squared = dot_product(abs_func, abs_func, -np.pi, np.pi)  # Вычисляем квадрат нормы функции
    a_coeffs = [a(i, abs_func, -np.pi, 2 * np.pi) for i in range(0, Config.N + 1)]  # Вычисляем коэффициенты a_n
    b_coeffs = [b(i, abs_func, -np.pi, 2 * np.pi) for i in range(0, Config.N + 1)]  # Вычисляем коэффициенты b_n
    c_coeffs = [c(i, abs_func, -np.pi, 2 * np.pi) for i in range(-Config.N, Config.N + 1)]  # Вычисляем коэффициенты c_n
    # Считаем суммы квадратов норм коэффициентов Фурье
    ab_sum = np.pi * (a_coeffs[0] ** 2 / 2 + sum([a_coeffs[i] ** 2 + b_coeffs[i] ** 2 for i in range(1, Config.N + 1)]))
    c_sum = 2 * np.pi * sum(abs(c_coeffs[i]) ** 2 for i in range(len(c_coeffs)))
    return abs(norm_squared - ab_sum), abs(norm_squared - c_sum)  # Возвращаем отклонения


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
    x = np.linspace(a, b, 10000)  # Генерируем точки на отрезке [a, b]
    dx = x[1] - x[0]  # Шаг интегрирования
    return np.dot(f(x), g(x)) * dx  # Возвращаем скалярное произведение


def a(n: int, func: Callable = f, start: float = -Config.T / 2, period: float = Config.T) -> np.ndarray:
    """
    Вычисляет коэффициент a_n для функции func на отрезке (start, start + period).
    При только заданном n вычисляет для функции и периода из конфига.

    :param n: Порядковый номер коэффициента.
    :type n: int
    :param func: Функция, для которой вычисляется коэффициент. По умолчанию используется функция из конфига.
    :type func: Callable, optional
    :param start: Начальное отрезка. По умолчанию равно -T / 2. (T берётся из конфига)
    :type start: float, optional
    :param period: Период функции и длина отрезка. По умолчанию равен T. (T берётся из конфига)
    :type period: float, optional
    :return: Значение коэффициента a_n.
    :rtype: np.ndarray
    """
    return 2 / period * dot_product(func, lambda t: np.cos(w(period) * n * t), start, start + period)


def b(n: int, func: Callable = f, start: float = -Config.T / 2, period: float = Config.T) -> np.ndarray:
    """
    Вычисляет коэффициент b_n для функции func на отрезке (start, start + period).
    При только заданном n вычисляет для функции и периода из конфига.

    :param n: Порядковый номер коэффициента.
    :type n: int
    :param func: Функция, для которой вычисляется коэффициент. По умолчанию используется функция из конфига.
    :type func: Callable, optional
    :param start: Начальное отрезка. По умолчанию равно -T / 2. (T берётся из конфига)
    :type start: float, optional
    :param period: Период функции и длина отрезка. По умолчанию равен T. (T берётся из конфига)
    :type period: float, optional
    :return: Значение коэффициента b_n.
    :rtype: np.ndarray
    """
    return 2 / period * dot_product(func, lambda t: np.sin(w(period) * n * t), start, start + period)


def c(n, func: Callable = f, start: float = -Config.T / 2, period: float = Config.T) -> np.ndarray:
    """
    Вычисляет коэффициент c_n для функции func на отрезке (start, start + period).
    При только заданном n вычисляет для функции и периода из конфига.

    :param n: Порядковый номер коэффициента.
    :type n: int
    :param func: Функция, для которой вычисляется коэффициент. По умолчанию используется функция из конфига.
    :type func: Callable, optional
    :param start: Начальное отрезка. По умолчанию равно -T / 2. (T берётся из конфига)
    :type start: float, optional
    :param period: Период функции и длина отрезка. По умолчанию равен T. (T берётся из конфига)
    :type period: float, optional
    :return: Значение коэффициента c_n.
    :rtype: np.ndarray
    """
    return 1 / period * dot_product(func, lambda t: np.exp(-1j * w(period) * n * t), start, start + period)


def calculate_fourier_real_graph(t: NDArray[np.floating[Any]], n: int):
    """
    Вычисляет значение графика тригонометрического ряда Фурье в точке x при n коэффициентах разложения.

    :param t: Значения точек x, для которых вычисляется значения y на графике.
    :type t: NDArray[np.floating[Any]]
    :param n: Количество коэффициентов разложения.
    :type n: int
    :return: Значения y тригонометрического ряда Фурье в точках x.
    :rtype: np.ndarray
    """
    alpha = w(Config.T) * t
    return a(0) / 2 + sum(a(i) * np.cos(alpha * i) + b(i) * np.sin(alpha * i) for i in range(1, n + 1))


def calculate_fourier_complex_graph(t: NDArray[np.floating[Any]], n: int):
    """
    Вычисляет значение графика экспоненциального ряда Фурье в точке t при n коэффициентах разложения.

    :param t: Значения точек x, для которых вычисляется значения y на графике.
    :type t: NDArray[np.floating[Any]]
    :param n: Количество коэффициентов разложения.
    :type n: int
    :return: Значения y экспоненциального ряда Фурье в точках x.
    :rtype: np.ndarray
    """
    return sum(c(i) * np.exp(1j * w(Config.T) * i * t) for i in range(-n, n + 1))


def main():
    if Config.TASK == 'parseval':
        return print(
            'Parseval deviation:\n'
            '| |f|^2 - sum(|a_i|^2 + |b_i|^2) | = {:.5f}\n'
            '| |f|^2 - sum(|c_i|^2) |           = {:.5f}'.format(*parseval_check())
        )
    if Config.TASK == 'coeffs':
        # Задаём период функции и порядковый номер коэффициента
        print('Оставьте поле ввода пустым, и программа выведет первые 6 коэффициентов Фурье, начиная с 0.')
        coef_num = int(data) if (data := input('Введите порядковый номер коэффициента: ')) else None

        # Выводим коэффициенты Фурье (либо первые 6, либо введённый пользователем)
        for n in range(coef_num or 0, (coef_num or 5) + 1):
            a_n, b_n, c_n, c_mn = a(n).round(3), b(n).round(3), c(n).round(3), c(-n).round(3)
            print(f'a_{n} = {a_n},\tb_{n} = {b_n},\tc_{n} = {c_n}  c_{-n} = {c_mn}')
        return

    x_values = np.linspace(min(Config.xticks), max(Config.xticks), 1000)  # Устанавливаем значения x для графика
    if Config.TASK == 'fourier':
        # Подсчитываем значения y графиков рядов Фурье
        F_N = calculate_fourier_real_graph(x_values, Config.N)
        print('Real calculated!')
        G_N = calculate_fourier_complex_graph(x_values, Config.N)
        print('Complex calculated!')

    # Настраиваем график, чтобы всё было по красоте
    fig, ax = plt.subplots()
    fig.set_size_inches(640 / fig.dpi, 358 / fig.dpi)
    pf_xlabel, pf_ylabel = {
        'complex': ('Re $f(t)$', 'Im $f(t)$'),
        'real': ('$t$', 'Re $f(t)$'),
        'imag': ('$t$', 'Im $f(t)$'),
    }.get(Config.PARAMETRIC_TO_DRAW, ('Re $f(t)$', 'Im $f(t)$'))
    ax.set_xlabel(pf_xlabel if Config.GRAPH_TYPE == 'parametric' else '$x$')
    ax.set_ylabel(pf_ylabel if Config.GRAPH_TYPE == 'parametric' else '$f(x)$')
    ax.set_xlim(min(0, min(Config.xticks)), max(0, max(Config.xticks)))
    ax.set_ylim(min(0, min(Config.yticks)), max(0, max(Config.yticks)))
    ax.axvline(0, color='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(Config.xticks)
    ax.set_xticklabels(Config.xlabels or Config.xticks)
    ax.set_yticks(Config.yticks)
    ax.set_yticklabels(Config.ylabels or Config.yticks)
    ax.grid()

    # Отрисовываем график основной функции
    match Config.GRAPH_TYPE:
        case 'square_wave':
            for i in range(len(LabFunctions.dots) - 1):
                sx, ex = LabFunctions.dots[i], LabFunctions.dots[i + 1]
                line = ax.plot([sx, ex], [calculate_graph(sx, 'start'), calculate_graph(ex, 'end')], color='#3454D1')
        case 'parametric':
            pf_xvals, pf_yvals = {
                'complex': (f(x_values).real, f(x_values).imag),
                'real': (x_values, f(x_values).real),
                'imag': (x_values, f(x_values).imag),
            }.get(Config.PARAMETRIC_TO_DRAW, (G_N.real, G_N.imag))
        case _:
            line = ax.plot(x_values, calculate_graph(x_values), color='#3454D1')

    line[0].set_label('$f(t)$' if Config.GRAPH_TYPE == 'parametric' else '$f(x)$')

    # Отрисовываем график рядов Фурье
    if Config.TASK == 'fourier':
        if Config.GRAPH_TYPE == 'parametric':
            pf_xvals, pf_yvals = {
                'complex': (G_N.real, G_N.imag),
                'real': (x_values, G_N.real),
                'imag': (x_values, G_N.imag),
            }.get(Config.PARAMETRIC_TO_DRAW, (G_N.real, G_N.imag))
            ax.plot(pf_xvals, pf_yvals, color='#D1345B', label=f'$G_{{{Config.N}}}(t)$')
        else:
            ax.plot(x_values, F_N, color='#34D1BF', linestyle='--', label=f'$F_{{{Config.N}}}(t)$')
            ax.plot(x_values, G_N, color='#D1345B', linestyle=':', linewidth=2, label=f'$G_{{{Config.N}}}(t)$')

    # Устанавливаем положение легенды, тонкие границы вокруг графика, включаем Tex и сохраняем график
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.rcParams.update({'text.usetex': True})
    plt.savefig(f'Im{"func" if Config.TASK == "empty" else Config.N}.png')
    print('Graph saved!')


if __name__ == '__main__':
    main()
