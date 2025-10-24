import numpy as np

def gaussian_kernel(size, sigma):
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    g = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    g /= np.sum(g)
    return g

for size in [3, 5, 7]:
    sigma = 5  
    kernel = gaussian_kernel(size, sigma)
    print(f"\nМатрица Гаусса {size}x{size}")
    print(np.round(kernel, 4))
    print(f"Сумма элементов: {np.sum(kernel)}")
