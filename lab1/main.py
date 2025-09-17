import cv2 as cv
# img = cv.imread("lab1\dog.jpg", cv.IMREAD_COLOR)
# resized = cv.resize(img, (800, 600))

# cv.namedWindow("output", cv.WINDOW_AUTOSIZE)
# cv.imshow("output", resized)
# cv.waitKey(0)

cap = cv.VideoCapture("lab1/dance.mov")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2Lab)   
    cv.imshow("Video", gray)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()