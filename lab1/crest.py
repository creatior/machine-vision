import cv2 as cv

# cap = cv.VideoCapture(1)

url = "http://192.168.1.67:4747/video"

cap = cv.VideoCapture(url)

if not cap.isOpened():
    print("Камера не найдена")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    b, g, r = frame[center_y, center_x]

    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }

    distances = {}
    for name, (cb, cg, cr) in colors.items():
        distances[name] = (int(r) - cr)**2 + (int(g) - cg)**2 + (int(b) - cb)**2

    nearest_color = min(distances, key=distances.get)
    color = colors[nearest_color]

    thickness = -1
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
