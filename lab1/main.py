import cv2 as cv
# img = cv.imread("lab1\dog.jpg", cv.IMREAD_COLOR)
# resized = cv.resize(img, (800, 600))

# cv.namedWindow("output", cv.WINDOW_AUTOSIZE)
# cv.imshow("output", resized)
# cv.waitKey(0)

# cap = cv.VideoCapture("lab1/dance.mov")

# fps = cap.get(cv.CAP_PROP_FPS)
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# print(f"Размер: {width}x{height}, FPS: {fps}")

# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter("lab1/dance_copy.mp4", fourcc, fps, (width, height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)

#     out.write(lab)

#     cv.imshow("Video", lab)

#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv.destroyAllWindows()

img = cv.imread("lab1/dog.jpg")

if img is None:
    print("Файл не найден")
    exit()

resized = cv.resize(img, (800, 600))

hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)

cv.imshow("Original (BGR)", resized)
cv.imshow("HSV", hsv)

cv.waitKey(0)
cv.destroyAllWindows()