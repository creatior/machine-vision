import cv2
import numpy as np

KERNEL_SIZE = 7
BLUR_PARAM = 20

def gaussian_kernel(size, sigma):
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    g = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    g /= np.sum(g)
    return g

def gaussian_filter(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    height, width = image.shape
    pad = size // 2

    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    result = np.zeros_like(image, dtype=float)

    for i in range(height):
        for j in range(width):
            region = padded[i:i+size, j:j+size]
            result[i, j] = np.sum(region * kernel)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

img = cv2.imread('images/mole.jpg', cv2.IMREAD_GRAYSCALE)
blurred = gaussian_filter(img, size=KERNEL_SIZE, sigma=BLUR_PARAM)

cv2.imshow('Исходное изображение', img)
cv2.imshow('После фильтра Гаусса', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
