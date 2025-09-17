import cv2 as cv

cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Камера не найдена")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    color = (0, 0, 255)  
    thickness = 2
    vert_height = 200
    hor_width = 200
    bar_thickness = 20

    top_left_vert = (center_x - bar_thickness//2, center_y - vert_height//2)
    bottom_right_vert = (center_x + bar_thickness//2, center_y + vert_height//2)
    cv.rectangle(frame, top_left_vert, bottom_right_vert, color, thickness)

    top_left_hor = (center_x - hor_width//2, center_y - bar_thickness//2)
    bottom_right_hor = (center_x + hor_width//2, center_y + bar_thickness//2)
    cv.rectangle(frame, top_left_hor, bottom_right_hor, color, thickness)

    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
