import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 140, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 140, 60])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_filtered = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    # Открытие = erode → dilate
    eroded = cv2.erode(mask, kernel, iterations=1)
    opening = cv2.dilate(eroded, kernel, iterations=1)

    # Закрытие = dilate → erode
    dilated = cv2.dilate(opening, kernel, iterations=1)
    closing = cv2.erode(dilated, kernel, iterations=1)

    red_filtered_morph = cv2.bitwise_and(frame, frame, mask=closing)

    # Вывод
    cv2.imshow("Original", frame)
    cv2.imshow("Filtered Red Only", red_filtered)
    cv2.imshow("Filtered Red Only Morphed", red_filtered_morph)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
