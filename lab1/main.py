import cv2 as cv
img = cv.imread("lab1\dog.jpg", cv.IMREAD_GRAYSCALE)
resized = cv.resize(img, (800, 600))

cv.namedWindow("output", cv.WINDOW_AUTOSIZE)
cv.imshow("output", resized)

cv.waitKey(0)