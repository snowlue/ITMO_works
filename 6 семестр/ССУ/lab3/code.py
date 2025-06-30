import numpy as np

import matplotlib.pyplot as plt

# Define the x range
x = np.linspace(0, 10, 1000)

# Define the rectangular waves
wave2 = np.where((x >= 4) & (x <= 7), 1, 0)

# Plot the waves
plt.plot(x, wave2, label='Магнит', color='blue')

# Set the graph boundaries
plt.xlim(0, max(x))
plt.ylim(0, 1.4)

# Add labels, legend, and grid
plt.xlabel('Расстояние')
plt.ylabel('Выходной сигнал')
plt.title('Магнитный датчик')
plt.legend()
plt.grid()

# Show the plot
plt.show()