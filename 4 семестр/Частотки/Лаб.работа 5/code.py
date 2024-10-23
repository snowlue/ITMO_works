import os
import timeit
from itertools import chain
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np

get_dt = lambda T, density: T / density

divs = lambda n: sorted(list(set(chain(*((d, n // d) for d in range(1, int(n**0.5) + 1) if n % d == 0)))))


def get_min_max(func: np.ndarray, step: int = 0, forced_min=None, forced_max=None) -> tuple[int, int, int]:
    """Возвращает минимальное и максимальное значения функции `func`."""
    min_ = floor(func.min()) if forced_min is None else forced_min
    max_ = ceil(func.max()) if forced_max is None else forced_max
    if not step:
        cur_divs = divs(max_ - min_)
        step = cur_divs[(len(cur_divs) - 1) // 2]
    return min_ - min_ % step, max_ - max_ % step + step, step


def plotter(
    plots: list[tuple[np.ndarray, np.ndarray, dict]],
    filename: str,
    labels: tuple[str, str],
    ticks: tuple[list[float] | range | None, list[float] | range | None] | None = None,
    tickslabels: tuple[list[str] | None, list[str] | None] | None = None,
    lims: tuple[tuple[float | None, float | None] | None, tuple[float | None, float | None] | None] | None = None,
    forced_fig=None,
    forced_ax=None,
) -> None:
    """Отрисовывает графики из plots в масштабе lims с осями labels и значениями ticks на них в файл filename (есть возможность задать кастомные tickslabels вместо ticks на осях)

    :param plots: список графиков (значения x, функция, доп.аргументы)
    :type plots: list[tuple[np.ndarray, np.vectorize, dict]]
    :param filename: название файла
    :type filename: str
    :param labels: названия осей
    :type labels: tuple[str, str]
    :param ticks: значения на осях
    :type ticks: tuple[list[float], list[float]]
    :param tickslabels: подписи значений на осях, заменяющие ticks
    :type tickslabels: tuple[list[str], list[str]], optional
    :param lims: крайние границы графика
    :type lims: tuple[tuple[float, float], tuple[float, float]]
    """
    if forced_fig and forced_ax:
        fig, ax = forced_fig, forced_ax
    else:
        fig, ax = plt.subplots()

    fig.set_size_inches(640 / fig.dpi, 358 / fig.dpi)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.axvline(0, color='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.7)

    if ticks:
        if ticks[0] is not None:
            ax.set_xlim(min(ticks[0]), max(ticks[0]))
            ax.set_xticks(ticks[0])
        if ticks[1] is not None:
            ax.set_ylim(min(ticks[1]), max(ticks[1]))
            ax.set_yticks(ticks[1])
    if tickslabels:
        if tickslabels[0] is not None:
            ax.set_xticklabels(tickslabels[0])
        if tickslabels[1] is not None:
            ax.set_yticklabels(tickslabels[1])
    if lims:
        if lims[0] is not None:
            ax.set_xlim(lims[0][0], lims[0][1])
        if lims[1] is not None:
            ax.set_ylim(lims[1][0], lims[1][1])
    ax.grid()
    for x_values, y_values, kwargs in plots:
        ax.plot(x_values, y_values, **kwargs)

    plt.legend(loc='lower right')
    plt.tight_layout()
    if not (forced_fig and forced_ax):
        plt.savefig(f'{filename}.png')
        plt.close(fig)


def square_wave_generator(a: float, t_1: float, t_2: float):
    """Генерирует функцию `f(t) = a if t_1 <= t <= t_2 else 0`. Возвращает её в векторизованном для numpy виде."""

    inner_basic_func = lambda t: a if t_1 <= t <= t_2 else 0

    return np.vectorize(inner_basic_func)


def dot_product(f: np.ndarray, g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Вычисляет скалярное произведение двух функций `f` и `g` на промежутке `x`."""
    dx = x[1] - x[0]
    return np.dot(f, g) * dx


fft = lambda X, V, func: np.array([dot_product(func, (lambda t: np.e ** (-1j * 2 * np.pi * v * t))(X), X) for v in V])
ifft = lambda X, V, image: np.array([dot_product(image, (lambda t: np.e ** (1j * 2 * np.pi * x * t))(V), V) for x in X])


def first_task_part1(T, density):
    os.makedirs('sources/first/part1', exist_ok=True)
    os.chdir('sources/first/part1')
    os.makedirs(f'{T=} {density=}', exist_ok=True)
    os.chdir(f'{T=} {density=}')

    t = np.linspace(-5, 5, density)
    v = np.linspace(-T, T, density)
    square_wave = square_wave_generator(1, -1 / 2, 1 / 2)(t)
    true_image = np.sinc(v)

    num_image = fft(t, v, square_wave)
    num_restored = ifft(t, v, num_image)

    plotter(  # 1_square_wave
        [(t, square_wave, {'color': '#3454D1', 'label': r'$\Pi(t)$'})],
        '../../1_square_wave',
        (r'$t$', r'$\Pi(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )
    plotter(  # 2_true_image
        [(v, true_image, {'color': '#34D1BF', 'label': r'$\hat{\Pi}(v)$'})],
        '../../2_true_image',
        (r'$v$', r'$\hat{\Pi}(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # 3_num_image
        [(v, num_image, {'color': '#7D33D1', 'label': r'$\hat{\Pi}_N(v)$'})],
        '3_num_image',
        (r'$v$', r'$\hat{\Pi}_N(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # 4_num_restored
        [(t, num_restored, {'color': '#6046d1', 'label': r'$\Pi_N(t)$'})],
        '4_num_restored',
        (r'$t$', r'$\Pi_N(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )
    plotter(  # 5_fft_cmp
        [
            (v, true_image, {'color': '#34D1BF', 'label': r'$\hat{\Pi}(v)$'}),
            (v, num_image, {'color': '#7D33D1', 'label': r'$\hat{\Pi}_N(v)$', 'linestyle': '--'}),
        ],
        '5_fft_cmp',
        (r'$v$', r'$\hat{\Pi}(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # 6_result_cmp
        [
            (t, square_wave, {'color': '#3454D1', 'label': r'$\Pi(t)$'}),
            (t, num_restored, {'color': '#6046d1', 'label': r'$\Pi_N(t)$', 'linestyle': '--'}),
        ],
        '6_ifft_cmp',
        (r'$t$', r'$\Pi(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )

    os.chdir('../../../..')


def first_task_part2():
    os.makedirs('sources/first/part2', exist_ok=True)
    os.chdir('sources/first/part2')

    density, T = 10000, 10
    t = np.linspace(-T / 2, T / 2, density)
    v_dft = np.fft.fftshift(np.fft.fftfreq(density, get_dt(T, density)))
    square_wave = square_wave_generator(1, -1 / 2, 1 / 2)(t)
    true_image = np.sinc(v_dft)

    dft_image = np.fft.fftshift(np.fft.fft(square_wave, norm='ortho'))
    dft_restored = np.fft.ifft(np.fft.ifftshift(dft_image), norm='ortho')

    plotter(  # 1_square_wave
        [(t, square_wave, {'color': '#3454D1', 'label': r'$\Pi(t)$'})],
        '../1_square_wave',
        (r'$t$', r'$\Pi(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )
    plotter(  # 2_true_image
        [(v_dft, true_image, {'color': '#34D1BF', 'label': r'$\hat{\Pi}(v)$'})],
        '../2_true_image',
        (r'$v$', r'$\hat{\Pi}(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # 3_dft_image
        [(v_dft, dft_image, {'color': '#7D33D1', 'label': r'$\hat{\Pi}_{dft}(v)$'})],
        '3_dft_image',
        (r'$v$', r'$\hat{\Pi}_{dft}(v)$'),
        ([i * 3 for i in range(-5, 6)], [i * 2 for i in range(-5, 6)]),
        lims=(None, (-11, 11)),
    )
    plotter(  # 4_dft_restored
        [(t, dft_restored, {'color': '#6046d1', 'label': r'$\Pi_{dft}(t)$'})],
        '4_dft_restored',
        (r'$t$', r'$\Pi_{dft}(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )

    os.chdir('../../..')


def first_task_part3():
    os.makedirs('sources/first/part3', exist_ok=True)
    os.chdir('sources/first/part3')

    density, T = 10000, 10
    t = np.linspace(-T / 2, T / 2, density)
    v_dft = np.fft.fftshift(np.fft.fftfreq(density, get_dt(T, density)))
    square_wave = square_wave_generator(1, -1 / 2, 1 / 2)(t)
    true_image = np.sinc(v_dft)

    dft_image = np.fft.fftshift(np.fft.fft(square_wave))
    dt = t[1] - t[0]
    dft_continuous_image = dft_image * dt * np.exp(-2j * np.pi * v_dft * t[0])
    dft_continuous_restored = np.fft.ifft(np.fft.ifftshift(dft_image))

    plotter(  # 1_square_wave
        [(t, square_wave, {'color': '#3454D1', 'label': r'$\Pi(t)$'})],
        '../1_square_wave',
        (r'$t$', r'$\Pi(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )
    plotter(  # 2_true_image
        [(v_dft, true_image, {'color': '#34D1BF', 'label': r'$\hat{\Pi}(v)$'})],
        '../2_true_image',
        (r'$v$', r'$\hat{\Pi}(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # 3_dft_continuous_image
        [(v_dft, dft_continuous_image, {'color': '#7D33D1', 'label': r'$\hat{\Pi}_{cdft}(v)$'})],
        '3_dft_continuous_image',
        (r'$v$', r'$\hat{\Pi}_{cdft}(v)$'),
        ([i * 2.5 for i in range(-4, 5)], [i / 5 for i in range(-1, 6)]),
        lims=(None, (-0.3, 1.1)),
    )
    plotter(  # dft_continuous_restored
        [(t, dft_continuous_restored, {'color': '#6046d1', 'label': r'$\Pi_{cdft}(t)$'})],
        '4_dft_continuous_restored',
        (r'$t$', r'$\Pi_{cdft}(t)$'),
        ([i / 2 for i in range(-5, 6)], [i / 10 for i in range(-5, 16, 5)]),
    )

    os.chdir('../../..')


def second_task(func, dt, B):
    os.makedirs('sources/second', exist_ok=True)
    os.chdir('sources/second')
    os.makedirs(f'{func.__name__} dt={round(dt, 3)} {B=}', exist_ok=True)
    os.chdir(f'{func.__name__} dt={round(dt, 3)} {B=}')

    t = np.linspace(-100, 100, 10000)
    t_sampled = np.arange(-100, 100, dt)
    y, y_sampled = func(t), func(t_sampled)

    v = np.linspace(-100, 100, 10000)
    if func.__name__ == 'sinc':
        image = fft(t, v, y)

    t_interp = np.linspace(-100, 100, 10000)
    y_interp = np.vectorize(lambda t: np.sum([func(n) * np.sinc(2 * B * (t - n)) for n in t_sampled]))(t_interp)
    if func.__name__ == 'sinc':
        interp_image = fft(t_interp, v, y_interp)

    min_, max_, step = get_min_max(y)
    plotter(  # 1_y
        [(t, y, {'color': '#3454D1', 'label': r'$y(t)$'})],
        f'../1_{func.__name__}_y',
        (r'$t$', r'$y(t)$'),
        (range(-10, 11, 2), range(min_, max_ + 1, step)),
    )
    plotter(  # 2_y_sampled
        [(t_sampled, y_sampled, {'color': '#D1345B', 'label': r'$y(t_n)$'})],
        '2_y_sampled',
        (r'$t$', r'$y(t)$'),
        (range(-10, 11, 2), range(min_, max_ + 1, step)),
    )
    plotter(  # 3_y_cmp(sampling)
        [
            (t, y, {'color': '#3454D1', 'label': r'$y(t)$'}),
            (t_sampled, y_sampled, {'color': '#D1345B', 'label': r'$y(t_n)$'}),
        ],
        '3_y_cmp(sampling)',
        (r'$t$', r'$y(t)$'),
        (range(-10, 11, 2), range(min_, max_ + 1, step)),
    )
    plotter(  # 4_y_interp
        [(t_interp, y_interp, {'color': '#34D1BF', 'label': r'$y_{int}(t)$'})],
        '4_y_interp',
        (r'$t$', r'$y_{int}(t)$'),
        (range(-10, 11, 2), range(min_, max_ + 1, step)),
    )
    plotter(  # 5_y_cmp(interpolation)
        [
            (t, y, {'color': '#3454D1', 'label': r'$y(t)$'}),
            (t_interp, y_interp, {'color': '#D1345B', 'label': r'$y_{int}(t)$', 'linestyle': '--'}),
        ],
        '5_y_cmp(interpolation)',
        (r'$t$', r'$y(t)$'),
        (range(-10, 11, 2), range(min_, max_ + 1, step)),
    )
    if func.__name__ == 'sinc':
        plotter(  # 6_image
            [(v, image, {'color': '#464CD1', 'label': r'$\hat{y}(v)$'})],
            '../6_sinc_image',
            (r'$v$', r'$\hat{y}(v)$'),
            (range(-6, 7), [i / 10 for i in range(-1, 7)]),
        )
        plotter(  # 7_interp_image
            [(v, interp_image, {'color': '#46AAC4', 'label': r'$\hat{y}_{int}(v)$'})],
            '7_interp_image',
            (r'$v$', r'$\hat{y}_{int}(v)$'),
            (range(-6, 7), [i / 10 for i in range(-1, 7)]),
        )

    os.chdir('../../../')


def main():
    exec_time = timeit.timeit(lambda: first_task_part1(20, 1000), number=1)
    print(f'trapz #1: {round(exec_time, 2)} s')
    exec_time = timeit.timeit(lambda: first_task_part1(90, 1000), number=1)
    print(f'trapz #2: {round(exec_time, 2)} s')
    exec_time = timeit.timeit(lambda: first_task_part1(90, 10000), number=1)
    print(f'trapz #3: {round(exec_time, 2)} s\n')

    exec_time = timeit.timeit(first_task_part2, number=1)
    print(f'dft: {round(exec_time, 2)} s')

    exec_time = timeit.timeit(first_task_part3, number=1)
    print(f'cdft: {round(exec_time, 2)} s')

    def sins(t):
        return np.sin(2 * t + 3) + 4 * np.sin(5 * t + 6)

    def sinc(t):
        return np.sinc(2 * t)

    second_task(sins, 1 / 8, 4)
    second_task(sins, 1 / 4, 4)
    second_task(sins, 1 / 4, 2)
    second_task(sinc, 1 / 4, 2)
    second_task(sinc, 1 / 2, 2)


if __name__ == '__main__':
    main()
