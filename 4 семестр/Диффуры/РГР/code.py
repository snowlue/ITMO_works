try:
    from phaseportrait import PhasePortrait2D
except ImportError:
    import os
    os.system('pip install phaseportrait')
    from phaseportrait import PhasePortrait2D

def df_1(x, y):
    x += 2
    y += 1
    return -2 * x - 8 * y, 4 * x - 8 * y + 1

def df_2(x, y):
    x -= 4
    y -= 2
    return 4 * x + 4 * y, -8 * x + 16 * y


def render_pp(df, x_center, y_center):
    pp = PhasePortrait2D(df, [min(x_center, y_center) - 3, max(x_center, y_center) + 3], numba=True, Title='', xlabel='x', ylabel='y')
    fig, ax = pp.plot()
    ax.set_xlim(x_center - 3, x_center + 3)
    ax.set_ylim(y_center - 3, y_center + 3)
    ax.set_xticks(range(x_center - 3, x_center + 4))
    ax.set_yticks(range(y_center - 3, y_center + 4))
    fig.savefig(f'{df.__name__}.png')


render_pp(df_1, -2, -1)
render_pp(df_2, 4, 2)
print('Done!')