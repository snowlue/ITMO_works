import numpy as np


def phi(t):
    if -2 <= t < -1.5:
        return -0.5
    if -1.5 <= t <= 0:
        return t + 1
    return 0


t_val = 0
while t_val <= 2.1:
    a = t_val - 2
    ans = phi(a)
    print(f't={t_val:.1f}, phi({a:.1f})={ans:.1f}, dx/dt(t)={-np.sign(ans)}')
    t_val += 0.1
