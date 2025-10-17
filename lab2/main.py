import cv2
import numpy as np

cap = cv2.VideoCapture(1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        moments = cv2.moments(mask)
        area = moments['m00']
        
        if area > 500:
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
            
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Area: {int(area)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Center: ({cx},{cy})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No significant red object", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No red object detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Original + Rectangle", frame)
    cv2.imshow("Red Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()