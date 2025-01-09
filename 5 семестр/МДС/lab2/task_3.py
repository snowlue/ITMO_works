import matplotlib.pyplot as plt
import numpy as np


sigma = np.linspace(-100, 100, 1000)
mu1 = 0
mu2 = 1 / 3

plt.plot(sigma, sigma / (np.abs(sigma) + 3), label=r'$\varphi(\sigma)=\sigma/(|\sigma|+3)$')
plt.plot(sigma, mu1 * sigma, label='0')
plt.plot(sigma, mu2 * sigma, label=r'$\mu_0=1/3$')
plt.title('Popov criterion')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\varphi(\sigma)$')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.legend()
plt.gcf().set_size_inches(6, 4)
plt.show()