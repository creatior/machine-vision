import cv2
import numpy as np

KERNEL_SIZE = 7
BLUR_PARAM = 20

img = cv2.imread('images/mole2.jpg', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), sigmaX=BLUR_PARAM)

cv2.imshow('Исходное изображение', img)
cv2.imshow('После фильтра Гаусса (cv2.GaussianBlur)', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
