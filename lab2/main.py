import cv2
import numpy as np

# Открываем камеру (0 - первая подключенная камера)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([40, 40, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("Original", frame)
    cv2.imshow("Threshold (Red mask)", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
