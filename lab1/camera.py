import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    cv2.imshow('Webcam', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
