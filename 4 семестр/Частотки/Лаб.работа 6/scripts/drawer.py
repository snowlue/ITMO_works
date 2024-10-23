import matplotlib.pyplot as plt
import numpy as np


def drawer(image: np.ndarray, filename: str, to_save: bool = True) -> None:
    plt.imshow(image)
    plt.axis('off')

    if to_save:
        return plt.imsave(f'{filename}.png', image, cmap='gray')
    plt.show()
