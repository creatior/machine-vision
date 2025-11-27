import cv2

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не найдено по указанному пути")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Черно-белое изображение", gray_image)
    cv2.waitKey(0)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    cv2.imshow("Размытие по Гауссу", blurred_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

process_image("cat.jpg")
