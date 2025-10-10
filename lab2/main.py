import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем изображение в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Диапазоны для красного цвета
    lower_red1 = np.array([0, 140, 60])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 140, 60])
    upper_red2 = np.array([180, 255, 255])

    # Создаем две маски и объединяем их
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Применяем маску к оригинальному изображению
    red_only = cv2.bitwise_and(frame, frame, mask=mask)

    # Показываем результат
    cv2.imshow("Original", frame)
    cv2.imshow("Threshold (Red mask)", mask)
    cv2.imshow("Filtered (Red Only)", red_only)

    # Нажми ESC для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
