import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numpy.fft import fft, fftfreq, ifft


def plotter(
    plots: list[tuple[np.ndarray, np.ndarray, dict]],
    filename: str,
    labels: tuple[str, str],
    lims: list[tuple[float, float]],
    ticks: tuple[list[float] | range, list[float] | range],
    tickslabels: tuple[list[str], list[str]] = None,  # type: ignore
) -> None:
    """Отрисовывает графики из plots в масштабе lims с осями labels и значениями ticks на них в файл filename

    :param plots: список графиков (значения x, функция, доп.аргументы)
    :type plots: list[tuple[np.ndarray, np.vectorize, dict]]
    :param filename: название файла
    :type filename: str
    :param labels: названия осей
    :type labels: tuple[str, str]
    :param lims: крайние границы графика
    :type lims: list[tuple[float, float]]
    :param ticks: значения на осях
    :type ticks: tuple[list[float], list[float]]
    :param tickslabels: подписи значений на осях, заменяющие ticks
    :type tickslabels: tuple[list[str], list[str]], optional
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(640 / fig.dpi, 358 / fig.dpi)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.axvline(0, color='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_xticks(ticks[0])
    ax.set_yticks(ticks[1])
    if tickslabels is not None:
        ax.set_xticklabels(tickslabels[0])
        ax.set_yticklabels(tickslabels[1])
    ax.grid()
    for x_values, y_values, kwargs in plots:
        ax.plot(x_values, y_values, **kwargs)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close(fig)


def basic_func_generator(a: float, t_1: float, t_2: float):
    """Генерирует функцию `f(t) = a if t_1 <= t <= t_2 else 0`. Возвращает её в векторизованном для numpy виде."""

    def inner_basic_func(t):
        return a if t_1 <= t <= t_2 else 0

    return np.vectorize(inner_basic_func)


def noisy_func_generator(g: np.ndarray, b: float, r: np.ndarray, c: float = 0, d: float = 0):
    """Возвращает зашумлённый вариант функции `g`."""

    def inner_noisy_func(t) -> np.ndarray:
        return g + b * (r - 0.5) + c * np.sin(d * t)

    return inner_noisy_func


def dot_product(f: np.ndarray, g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Вычисляет скалярное произведение двух функций `f` и `g` на промежутке `x`."""
    dx = x[1] - x[0]
    return np.dot(f, g) * dx


def fft_(func: np.ndarray):
    """Возвращает векторизованное для numpy преобразование Фурье для дальнейшей передачи в него частоты."""
    image = (
        lambda v: 1 / (np.sqrt(2 * np.pi)) * dot_product(func, (lambda t: np.e ** (-1j * v * t))(interval), interval)
    )
    return np.vectorize(image, otypes=[np.complex_])


def ifft_(func: np.ndarray):
    orig = lambda t: 1 / (np.sqrt(2 * np.pi)) * dot_product(func, (lambda v: np.e ** (1j * v * t))(ffreqs), ffreqs)
    return np.vectorize(orig)


def clipper(ctype: str, func: np.vectorize, limit: float):
    """Возвращает функцию, которая обрезает значения, передаваемые в функцию до `[-limit, limit]`."""
    match ctype:
        case 'low-pass':
            condition = lambda x: np.abs(x) >= limit
        case 'high-pass':
            condition = lambda x: np.abs(x) <= limit
        case 'band-stop':
            condition = lambda x: max(limit - 3, 0) <= np.abs(x) <= limit + 3
    return np.vectorize(lambda x_values: np.where(condition(x_values), 0, func(x_values)))


def audio_band_stop_clipper(X: np.ndarray, func: np.ndarray, limit: float, width: int):
    """Обрезает значения, передаваемые в функцию до `[-limit, limit]`."""
    return np.array(
        [
            0 if (limit - width <= X[i] <= limit + width or limit - width <= -X[i] <= limit + width) else func[i]
            for i in range(len(func))
        ]
    )


def generate_low_pass():
    """Low-pass filter image generator"""
    ffreqs = np.linspace(-15, 15, 1000)
    vs, bs = (3, 7, 13), (0.5, 1.5, 3)

    for b in bs:
        u = noisy_func_generator(g, b, rndm)(interval)
        plotter(
            [(interval, u, {'color': '#D1345B', 'label': '$u(t)$'})],
            f'sources/low-pass filter/noisy (b={b})',
            ('$t$', '$u(t)$'),
            [(-4, 7), (-2, 5)],
            (range(-4, 8), range(-2, 6)),
        )
        for v in vs:
            fourier_u = fft_(u)
            clipped_fourier_u = clipper('low-pass', fourier_u, v)(ffreqs)
            denoised_u = ifft_(clipped_fourier_u)(interval)
            fourier_denoised_u = fft_(denoised_u)(ffreqs)
            plotter(
                [
                    (
                        ffreqs,
                        np.abs(fourier_u(ffreqs)),
                        {'color': '#361665', 'label': r'$\hat{u}(v)$', 'linestyle': '--'},
                    ),
                    (ffreqs, np.abs(fourier_denoised_u), {'color': '#883aff', 'label': r'$\hat{u}_d(v)$'}),
                ],
                f'sources/low-pass filter/fourier (b={b}, v={v})',
                (r'$v$', r'$u(v)$'),
                [(-15, 15), (-1, 7)],
                (range(-15, 16, 3), range(-1, 8)),
            )
            plotter(
                [
                    (interval, g, {'color': '#3454D1', 'label': '$g(t)$'}),
                    (interval, denoised_u, {'color': '#34D1BF', 'label': '$u_d(t)$'}),
                ],
                f'sources/low-pass filter/denoised (b={b}, v={v})',
                ('$t$', '$f(t)$'),
                [(-4, 7), (-2, 5)],
                (range(-4, 8), range(-2, 6)),
            )


def generate_band_stop():
    """Band-stop filter image generator"""
    ffreqs = np.linspace(-20, 20, 1000)
    vs, bs, cs, ds = (5, 9, 16), (0, 1, 2), (0.8, 1, 1.5), (8, 10, 15)

    for b in bs:
        for c in cs:
            for d in ds:
                u = noisy_func_generator(g, b, rndm, c, d)(interval)
                plotter(
                    [(interval, u, {'color': '#D1345B', 'label': '$u(t)$'})],
                    f'sources/band-stop filter/noisy (b={b}, c={c}, d={d})',
                    ('$t$', '$u(t)$'),
                    [(-4, 7), (-2, 5)],
                    (range(-4, 8), range(-2, 6)),
                )
                for v in vs:
                    fourier_u = fft_(u)
                    clipped_fourier_u = clipper('band-stop', fourier_u, v)(ffreqs)
                    denoised_u = ifft_(clipped_fourier_u)(interval)
                    fourier_denoised_u = fft_(denoised_u)(ffreqs)
                    plotter(
                        [
                            (
                                ffreqs,
                                np.abs(fourier_u(ffreqs)),
                                {'color': '#361665', 'label': r'$\hat{u}(v)$', 'linestyle': '--'},
                            ),
                            (ffreqs, np.abs(fourier_denoised_u), {'color': '#883aff', 'label': r'$\hat{u}_d(v)$'}),
                        ],
                        f'sources/band-stop filter/fourier (b={b}, c={c}, d={d}, v={v})',
                        (r'$v$', r'$u(v)$'),
                        [(-20, 20), (-1, 7)],
                        (range(-20, 21, 4), range(-1, 8)),
                    )
                    plotter(
                        [
                            (interval, g, {'color': '#3454D1', 'label': '$g(t)$'}),
                            (interval, denoised_u, {'color': '#34D1BF', 'label': '$u_d(t)$'}),
                        ],
                        f'sources/band-stop filter/denoised (b={b}, c={c}, d={d}, v={v})',
                        ('$t$', '$f(t)$'),
                        [(-4, 7), (-2, 5)],
                        (range(-4, 8), range(-2, 6)),
                    )


def generate_high_pass():
    """High-pass filter image generator"""
    ffreqs = np.linspace(-20, 20, 1000)
    vs, bs, cs, ds = (3, 6, 11), (0, 1, 2), (0.8, 1, 1.5), (8, 10, 15)

    for b in bs:
        for c in cs:
            for d in ds:
                u = noisy_func_generator(g, b, rndm, c, d)(interval)
                plotter(
                    [(interval, u, {'color': '#D1345B', 'label': '$u(t)$'})],
                    f'sources/high-pass filter/noisy (b={b}, c={c}, d={d})',
                    ('$t$', '$u(t)$'),
                    [(-4, 7), (-2, 5)],
                    (range(-4, 8), range(-2, 6)),
                )
                for v in vs:
                    fourier_u = fft_(u)
                    clipped_fourier_u = clipper('high-pass', fourier_u, v)(ffreqs)
                    denoised_u = ifft_(clipped_fourier_u)(interval)
                    fourier_denoised_u = fft_(denoised_u)(ffreqs)
                    plotter(
                        [
                            (
                                ffreqs,
                                np.abs(fourier_u(ffreqs)),
                                {'color': '#361665', 'label': r'$\hat{u}(v)$', 'linestyle': '--'},
                            ),
                            (ffreqs, np.abs(fourier_denoised_u), {'color': '#883aff', 'label': r'$\hat{u}_d(v)$'}),
                        ],
                        f'sources/high-pass filter/fourier (b={b}, c={c}, d={d}, v={v})',
                        (r'$v$', r'$u(v)$'),
                        [(-20, 20), (-1, 7)],
                        (range(-20, 21, 4), range(-1, 8)),
                    )
                    plotter(
                        [
                            (interval, g, {'color': '#3454D1', 'label': '$g(t)$'}),
                            (interval, denoised_u, {'color': '#34D1BF', 'label': '$u_d(t)$'}),
                        ],
                        f'sources/high-pass filter/denoised (b={b}, c={c}, d={d}, v={v})',
                        ('$t$', '$f(t)$'),
                        [(-4, 7), (-2, 5)],
                        (range(-4, 8), range(-2, 6)),
                    )


def generate_audio():
    """Audio filtering image generator"""
    samples, sr = librosa.load('sources/audio/MUHA.wav')
    plotter(
        [(np.linspace(0, len(samples) / sr, len(samples)), samples, {'color': '#D1345B', 'label': '$f(t)$'})],
        'sources/audio/MUHA',
        ('$t$', '$f(t)$'),
        [(0, len(samples) / sr), (-0.75, 0.75)],
        (range(0, int(len(samples) / sr) + 1), [i / 4 for i in range(-3, 4)]),
    )  # исходный сигнал
    fourier_samples = fft(samples)
    V = fftfreq(len(samples), 1 / sr)
    clipped_fourier = audio_band_stop_clipper(V, fourier_samples, 150, 150)
    clipped_fourier = audio_band_stop_clipper(V, clipped_fourier, 400, 50)
    plotter(
        [
            (V, abs(fourier_samples), {'color': '#361665', 'label': r'$\hat{f}(v)$', 'linestyle': '--'}),
            (V, abs(clipped_fourier), {'color': '#883aff', 'label': r'$\hat{f}_d(v)$'}),
        ],
        'sources/audio/fourier_MUHA',
        ('$v$', r'$\hat{f}(v)$'),
        [(-700, 700), (0, 7000)],
        ticks=(range(-700, 701, 100), range(0, 7001, 1000)),
    )  # Фурье-образы исходного и отфильтрованного сигналов
    restored = ifft(clipped_fourier).real
    plotter(
        [(np.linspace(0, len(samples) / sr, len(samples)), restored, {'color': '#34D1BF', 'label': '$f_d(t)$'})],
        'sources/audio/MUHA_denoised',
        ('$t$', '$f(t)$'),
        [(0, len(samples) / sr), (-0.75, 0.75)],
        (range(0, int(len(samples) / sr) + 1), [i / 4 for i in range(-3, 4)]),
    )  # восстановленный сигнал
    sf.write('sources/audio/MUHA_denoised.wav', restored, sr, 'PCM_24')


# Basic function
T, dt = 14, 0.01
interval = np.linspace(-T / 2, T / 2, int(T / dt))
g: np.ndarray = basic_func_generator(3, -1, 4)(interval)
plotter(
    [(interval, g, {'color': '#3454D1', 'label': '$g(t)$'})],
    'sources/basic_function',
    ('$t$', '$g(t)$'),
    [(-4, 7), (-2, 5)],
    (range(-4, 8), range(-2, 6)),
)

# Preparing for filtering
rndm = np.random.default_rng(1).random(interval.size)
ffreqs = np.linspace(-15, 15, 1000)
for folder_name in 'sources/low-pass filter', 'sources/band-stop filter', 'sources/high-pass filter':
    os.makedirs(folder_name, exist_ok=True)

# generate_low_pass()  # Low-pass filter
# generate_band_stop()  # Band-stop filter
# generate_high_pass()  # High-pass filter
generate_audio()  # Audio filtering
