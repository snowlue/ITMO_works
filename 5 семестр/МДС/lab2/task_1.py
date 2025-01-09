import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sat(par):
    if np.abs(par) < 1:
        return par
    return np.sign(par)


def nonlinear_system(t, y):
    x1, x2 = y
    dx1_dt = -x1 - 3 * x2
    dx2_dt = -x1 - 4 * x2 - 2 * sat(x2)
    return [dx1_dt, dx2_dt]


def linearized_system(t, y):
    x1, x2 = y
    dx1_dt = -x1 - 3 * x2
    dx2_dt = -x1 - 6 * x2
    return [dx1_dt, dx2_dt]


x0 = [5.0, 10.0]

t_span = (0, 12)
t_eval = np.linspace(t_span[0], t_span[1], 500)

src_sol = solve_ivp(nonlinear_system, t_span, x0, method='RK45', t_eval=t_eval)
lin_sol = solve_ivp(linearized_system, t_span, x0, method='RK45', t_eval=t_eval)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(src_sol.t, src_sol.y[0], label='x(t)', color='b')
axes[0].plot(src_sol.t, src_sol.y[1], label='y(t)', color='r', linestyle='--')
axes[0].set_title('Source system')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('State variable')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(lin_sol.t, lin_sol.y[0], label='x(t)', color='b')
axes[1].plot(lin_sol.t, lin_sol.y[1], label='y(t)', color='r', linestyle='--')
axes[1].set_title('Linearized system')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('State variable')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()