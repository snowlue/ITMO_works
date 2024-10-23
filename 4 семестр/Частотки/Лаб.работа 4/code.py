import os
from datetime import datetime
from itertools import chain
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqs_zpk, lsim, tf2zpk

density = 5000
get_dt = lambda T: T / density
get_linspace = lambda t_1, t_2: np.linspace(t_1, t_2, density)

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


def generate_noisy_func(f: np.ndarray, t: np.ndarray, b: float, c: float, d: float) -> np.ndarray:
    """Возвращает зашумлённый вариант функции `f`."""
    rndm = np.random.default_rng(1).random(t.size)
    return f + b * (rndm - 0.5) + c * np.sin(d * t)


def differentiate(func: np.ndarray, T: float):
    return np.array([0 if i == 0 else (func[i] - func[i - 1]) / get_dt(T) for i in range(len(func))])


def dot_product(f: np.ndarray, g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Вычисляет скалярное произведение двух функций `f` и `g` на промежутке `x`."""
    dx = x[1] - x[0]
    return np.dot(f, g) * dx


fft = lambda X, V, func: np.array(
    [1 / (np.sqrt(2 * np.pi)) * dot_product(func, (lambda t: np.e ** (-1j * v * t))(X), X) for v in V]
)
ifft = lambda X, V, image: np.array(
    [1 / (np.sqrt(2 * np.pi)) * dot_product(image, (lambda t: np.e ** (1j * x * t))(V), V) for x in X]
)
first_order_filter = lambda T: tf2zpk([0, 1], [T, 1])
second_order_filter = lambda T1, T2, T3: tf2zpk([T1**2, 2 * T1, 1], [T2 * T3, T2 + T3, 1])


def first_task():
    os.makedirs('sources/first', exist_ok=True)
    os.chdir('sources/first')

    T = 50
    t = get_linspace(-T / 2, T / 2)

    y = np.sin(t)

    signal = generate_noisy_func(y, t, 0.1, 0, 0)
    numerical_diff = differentiate(signal, T)

    fft_signal = fft(t, t, signal)
    diff_fft_signal = fft_signal * 1j * t
    spectral_diff = ifft(t, t, diff_fft_signal)

    wider_t = get_linspace(-2 * T, 2 * T)
    wider_fft_signal = fft(wider_t, wider_t, signal)
    diff_wider_fft_signal = wider_fft_signal * 1j * wider_t
    wider_spectral_diff = ifft(wider_t, wider_t, diff_wider_fft_signal)

    plotter(  # 1_signal
        [
            (
                t,
                signal,
                {'color': '#3454D1', 'label': r'$\sin(t)+0.1\left(\text{rand}\left(\text{size}(t)\right)-0.5\right)$'},
            )
        ],
        '1_signal',
        (r'$t$', r'$y(t)$'),
        (list(map(lambda x: x * np.pi, range(-4, 5))), [-1, -0.5, 0, 0.5, 1]),
        (
            [r'$-4\pi$', r'$-3\pi$', r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'],
            ['-1', '-0.5', '0', '0.5', '1'],
        ),
        (None, (-1.1, 1.1)),
    )
    plotter(  # 2_numerical_diff
        [(t, numerical_diff, {'color': '#1A2B68', 'label': r"$y'_{\text{числ}}(t)$"})],
        '2_numerical_diff',
        (r'$t$', r"$y'_{\text{числ}}(t)$"),
        (list(map(lambda x: x * np.pi, range(-4, 5))), range(-12, 13, 3)),
        ([r'$-4\pi$', r'$-3\pi$', r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], None),
    )
    plotter(  # 3_fft
        [
            (t, fft_signal.real, {'color': '#34D1BF', 'label': r'$Re\ \hat{y}(t)$'}),
            (t, fft_signal.imag, {'color': '#7D33D1', 'label': r'$Im\ \hat{y}(t)$'}),
        ],
        '3_fft',
        (r'$t$', r'$\hat{y}(v)$'),
        (
            range(-4, 5),
            range(-10, 11, 2),
        ),
        lims=(None, (-11, 11)),
    )
    plotter(  # 4_spectral_fft
        [
            (t, diff_fft_signal.real, {'color': '#34D1BF', 'label': r"$Re\ \hat{y}'_S(t)$"}),
            (t, diff_fft_signal.imag, {'color': '#7D33D1', 'label': r"$Im\ \hat{y}'_S(t)$"}),
        ],
        '4_spectral_fft',
        (r'$t$', r"$\hat{y}'_S(v)$"),
        (
            range(-4, 5),
            range(-10, 11, 2),
        ),
        lims=(None, (-11, 11)),
    )
    plotter(  # 5_diff_cmp
        [
            (t, spectral_diff, {'color': '#D1345B', 'label': r"$y'_S(t)$"}),
            (t, np.cos(t), {'color': '#34D1BF', 'label': r'$\cos(t)$'}),
        ],
        '5_diff_cmp',
        (r'$t$', r'$f(t)$'),
        (list(map(lambda x: x * np.pi, range(-4, 5))), [-1, -0.5, 0, 0.5, 1]),
        (
            [r'$-4\pi$', r'$-3\pi$', r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'],
            ['-1', '-0.5', '0', '0.5', '1'],
        ),
        (None, (-1.2, 1.2)),
    )
    plotter(  # 6_wider_fft
        [
            (wider_t, wider_fft_signal.real, {'color': '#34D1BF', 'label': r'$Re\ \hat{y}(t)$'}),
            (wider_t, wider_fft_signal.imag, {'color': '#7D33D1', 'label': r'$Im\ \hat{y}(t)$'}),
        ],
        '6_wider_fft',
        (r'$t$', r'$\hat{y}(v)$'),
        (range(-4, 5), None),
    )
    plotter(  # 7_wider_spectral_fft
        [
            (wider_t, diff_wider_fft_signal.real, {'color': '#34D1BF', 'label': r"$Re\ \hat{y}'_S(t)$"}),
            (wider_t, diff_wider_fft_signal.imag, {'color': '#7D33D1', 'label': r"$Im\ \hat{y}'_S(t)$"}),
        ],
        '7_wider_spectral_fft',
        (r'$t$', r"$\hat{y}'_S(v)$"),
        (range(-2 * T, 2 * T + 1, 20), None),
    )
    plotter(  # 8_wider_diff_cmp
        [
            (wider_t, wider_spectral_diff, {'color': '#D1345B', 'label': r"$y'_S(t)$"}),
            (wider_t, np.cos(wider_t), {'color': '#34D1BF', 'label': r'$\cos(t)$'}),
        ],
        '8_wider_diff_cmp',
        (r'$t$', r'$f(t)$'),
        (list(map(lambda x: x * np.pi, range(-4, 5))), range(-3, 4)),
        ([r'$-4\pi$', r'$-3\pi$', r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], None),
    )

    os.chdir('../..')


def second_task_part1(a: int, T_filter: float):
    os.makedirs('sources/second/part1', exist_ok=True)
    os.chdir('sources/second/part1')
    os.makedirs(f'{a=} T={T_filter}', exist_ok=True)
    os.chdir(f'{a=} T={T_filter}')

    T = 10
    t = get_linspace(0, T)
    v = get_linspace(-4 * T, 4 * T)
    filter = first_order_filter(T_filter)

    signal = square_wave_generator(a, 2, 5)(t)
    noisy_signal = generate_noisy_func(signal, t, 1, 0, 0)
    filtered_signal = lsim(filter, noisy_signal, t)[1]

    fft_signal = np.abs(fft(t, v, signal))
    fft_noisy_signal = np.abs(fft(t, v, noisy_signal))
    fft_filtered_signal = np.abs(fft(t, v, filtered_signal))

    z, p, k = filter
    w, h = freqs_zpk(z, p, k, worN=get_linspace(0, 2 * T))

    _, max_, step = get_min_max(noisy_signal, forced_min=-1, forced_max=a + 1)
    plotter(  # 1_signal_cmp
        [
            (t, noisy_signal, {'color': '#D1345B', 'label': r'$u_N(t)$'}),
            (t, signal, {'color': '#3454D1', 'label': r'$u(t)$'}),
            (t, filtered_signal, {'color': '#34D1BF', 'label': r'$u_F(t)$'}),
        ],
        '1_signal_cmp',
        (r'$t$', r'$u(t)$'),
        (range(0, 11), range(-1, max_, step)),
    )
    plotter(  # 2_fft_cmp
        [
            (v, fft_signal, {'color': '#3454D1', 'label': r'$|\hat{u}(v)|$'}),
            (v, fft_noisy_signal, {'color': '#D1345B', 'label': r'$|\hat{u}_N(t)|$', 'linestyle': '--'}),
            (v, fft_filtered_signal, {'color': '#34D1BF', 'label': r'$|\hat{u}_F(v)|$'}),
        ],
        '2_fft_cmp',
        (r'$v$', r'$|\hat{u}(v)|$'),
        (range(*get_min_max(v)), range(*get_min_max(fft_noisy_signal))),
    )
    index = np.absolute(abs(h) - 1 / np.sqrt(2)).argmin()
    plotter(  # 3_filter
        [
            (w, abs(h), {'color': '#7D33D1', 'label': r'$|W(i\omega)|$'}),
            (
                np.linspace(min(w), w[index], 100),
                np.array([1 / np.sqrt(2)] * 100),
                {'linestyle': '--', 'color': '#ffe100', 'label': r'$\omega_c$'},
            ),
            (np.array([w[index]] * 100), np.linspace(0, abs(h)[index], 100), {'linestyle': '--', 'color': '#ffe100'}),
        ],
        '3_filter',
        (r'$\omega$', r'$|W(i\omega)|$'),
        (range(0, 21, 2), [i / 10 for i in range(0, 11, 2)]),
    )

    os.chdir('../../../..')


def second_task_part2(c: float, d: float, T1: float, T2: float, T3: float):
    os.makedirs('sources/second/part2', exist_ok=True)
    os.chdir('sources/second/part2')
    os.makedirs(f'{c=}_{d=} {T1=}_T2={round(T2, 3)}_T3={round(T3, 3)}', exist_ok=True)
    os.chdir(f'{c=}_{d=} {T1=}_T2={round(T2, 3)}_T3={round(T3, 3)}')

    T = 10
    t = get_linspace(0, T)
    v = get_linspace(-4 * T, 4 * T)
    filter = second_order_filter(T1, T2, T3)

    signal = square_wave_generator(5, 2, 5)(t)
    noisy_signal = generate_noisy_func(signal, t, 0, c, d)
    filtered_signal = lsim(filter, noisy_signal, t)[1]

    fft_signal = np.abs(fft(t, v, signal))
    fft_noisy_signal = np.abs(fft(t, v, noisy_signal))
    fft_filtered_signal = np.abs(fft(t, v, filtered_signal))

    z, p, k = filter
    w, h = freqs_zpk(z, p, k, worN=get_linspace(0, 2 * T))

    min_, max_, step = get_min_max(noisy_signal)
    plotter(  # 1_signal_cmp
        [
            (t, noisy_signal, {'color': '#D1345B', 'label': r'$u_N(t)$'}),
            (t, signal, {'color': '#3454D1', 'label': r'$u(t)$'}),
            (t, filtered_signal, {'color': '#34D1BF', 'label': r'$u_F(t)$'}),
        ],
        '1_signal_cmp',
        (r'$t$', r'$u(t)$'),
        (range(0, 11), range(min_, max_, step)),
    )
    plotter(  # 2_fft_cmp
        [
            (v, fft_signal, {'color': '#3454D1', 'label': r'$|\hat{u}(v)|$'}),
            (v, fft_noisy_signal, {'color': '#D1345B', 'label': r'$|\hat{u}_N(t)|$', 'linestyle': '--'}),
            (v, fft_filtered_signal, {'color': '#34D1BF', 'label': r'$|\hat{u}_F(v)|$'}),
        ],
        '2_fft_cmp',
        (r'$v$', r'$|\hat{u}(v)|$'),
        (range(*get_min_max(v)), range(*get_min_max(fft_noisy_signal))),
    )
    index = np.absolute(abs(h) - 1 / np.sqrt(2)).argmin()
    plotter(  # 3_filter
        [
            (w, abs(h), {'color': '#7D33D1', 'label': r'$|W(i\omega)|$'}),
            (
                np.linspace(min(w), w[index], 100),
                np.array([1 / np.sqrt(2)] * 100),
                {'linestyle': '--', 'color': '#ffe100', 'label': r'$\omega_c$'},
            ),
            (np.array([w[index]] * 100), np.linspace(0, abs(h)[index], 100), {'linestyle': '--', 'color': '#ffe100'}),
        ],
        '3_filter',
        (r'$\omega$', r'$|W(i\omega)|$'),
        (range(0, 21, 2), [i / 10 for i in range(0, 11, 2)]),
    )

    os.chdir('../../../..')


def third_task(T):
    os.makedirs('sources/third', exist_ok=True)
    os.chdir('sources/third')

    dates, prices = [], []
    with open('../VKCO_200702_231229.csv', 'r') as file:
        for line in file:
            date_str, _, _, _, _, close, _ = line.split(';')
            date = datetime.strptime(date_str, '%d%m%y')
            dates, prices = dates + [date], prices + [float(close)]
    dates, prices = np.array(dates), np.array(prices)

    t = np.linspace(0, len(prices), len(prices))
    filter = first_order_filter(T)
    filtered_prices = lsim(filter, prices, t, prices[0] * T)[1]

    T_to_str = {1: '1 день', 7: '1 неделю', 30: '1 месяц', 90: '3 месяца', 365: '1 год'}
    plotter(
        [
            (t, prices, {'color': '#000', 'label': 'Цена'}),
            (t, filtered_prices, {'color': '#34D1BF', 'label': f'Скользящее среднее за {T_to_str.get(T, T)}'}),
        ],
        f'moving_average_{T}',
        ('День (02.07.2020 - 29.12.2023)', 'Цена'),
    )

    os.chdir('../..')


def main():
    first_task()
    second_task_part1(5, 0.5)
    second_task_part1(5, 0.1)
    second_task_part1(20, 0.1)
    second_task_part2(0.5, 18, 10**-8, 2 / 18, 2 / 18)
    second_task_part2(0.5, 30, 10**-8, 2 / 18, 2 / 18)
    second_task_part2(3, 30, 10**-8, 2 / 18, 2 / 18)
    third_task(1)
    third_task(7)
    third_task(30)
    third_task(90)
    third_task(365)


if __name__ == '__main__':
    main()
