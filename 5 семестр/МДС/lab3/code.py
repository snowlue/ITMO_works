import os
try:
    from phaseportrait import PhasePortrait2D
except ImportError:
    os.system('pip install phaseportrait')
    from phaseportrait import PhasePortrait2D

r = 0  # значение r можно задать здесь
def df(x1, x2):
    return x2, -x1 - (x1 ** 2 - r) * x2


def render_pp(df, x_center, y_center):
    pp = PhasePortrait2D(df, [min(x_center, y_center) - 3, max(x_center, y_center) + 3], numba=True, Title='', xlabel='$x_1$', ylabel='$x_2$')
    fig, ax = pp.plot()
    ax.set_xlim(x_center - 3, x_center + 3)
    ax.set_ylim(y_center - 3, y_center + 3)
    ax.set_xticks(range(x_center - 3, x_center + 4))
    ax.set_yticks(range(y_center - 3, y_center + 4))
    fig.savefig(f'sources/{df.__name__}.png')

os.makedirs('sources', exist_ok=True)
render_pp(df, 0, 0)
print('Done!')
