import cv2
import numpy as np

def get_quantized_angle(grad_x, grad_y, tg):
    quantized_angle = np.zeros_like(grad_x, dtype=np.uint8)

    mask0 = ((grad_x > 0) & (grad_y < 0) & (tg < -2.414)) | ((grad_x < 0) & (grad_y < 0) & (tg > 2.414))
    quantized_angle[mask0] = 0

    mask1 = (grad_x > 0) & (grad_y < 0) & (tg >= -2.414) & (tg < -0.414)
    quantized_angle[mask1] = 1

    mask2 = ((grad_x > 0) & (grad_y < 0) & (tg >= -0.414)) | ((grad_x > 0) & (grad_y > 0) & (tg < 0.414))
    quantized_angle[mask2] = 2

    mask3 = (grad_x > 0) & (grad_y > 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask3] = 3

    mask4 = ((grad_x > 0) & (grad_y > 0) & (tg >= 2.414)) | ((grad_x < 0) & (grad_y > 0) & (tg <= -2.414))
    quantized_angle[mask4] = 4

    mask5 = (grad_x < 0) & (grad_y > 0) & (tg > -2.414) & (tg <= -0.414)
    quantized_angle[mask5] = 5

    mask6 = ((grad_x < 0) & (grad_y > 0) & (tg > -0.414)) | ((grad_x < 0) & (grad_y < 0) & (tg < 0.414))
    quantized_angle[mask6] = 6

    mask7 = (grad_x < 0) & (grad_y < 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask7] = 7

    return quantized_angle

def process_image_with_gradients(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не найдено по указанному пути")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    img = blurred.astype(np.float64)

    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_kernel_x)
            grad_y[i, j] = np.sum(region * sobel_kernel_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_x_safe = np.where(grad_x == 0, 1e-6, grad_x)
    tg = grad_y / grad_x_safe
    
    quantized_angle = get_quantized_angle(grad_x, grad_y, tg)

    nms = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = quantized_angle[i, j]
            magnitude_value = magnitude[i, j]

            if direction in [0, 4]:
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif direction in [1, 5]:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif direction in [2, 6]:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            else:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            
            if magnitude_value > max(neighbors):
                nms[i, j] = 255
            else:
                nms[i, j] = 0

    cv2.imshow("Original", image)
    cv2.imshow("Gray + Blurred", blurred)
    cv2.imshow("NMS", nms)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return magnitude, quantized_angle

mg, qa = process_image_with_gradients("dog.jpg")
print(f"МАТРИЦА ДЛИН ГРАДИЕНТОВ:\n{mg}\nМАТРИЦА УГЛОВ ГРАДИЕНТОВ:\n{qa}")