import cv2
import numpy as np

img = cv2.imread("rj.jpg")


x, y, w, h = 500, 300, 800, 600
roi = img[y:y+h, x:x+w]

# blurred_roi = cv2.GaussianBlur(roi,(15,15),50)


blur = cv2.bilateralFilter(img, 9, 75, 75)


# img[y:y+h, x:x+w] = blurred_roi

# blur = cv2.GaussianBlur(img,(3,3),1)

cv2.imshow("Original", img)
cv2.imshow("Blur", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


