import cv2
import numpy as np

img = cv2.imread("rj.jpg")

if img is None:
    print("Image not found")
    exit()


blur = cv2.GaussianBlur(img,(7,7),1)



# Threshold1	Threshold2	            Effect
# Low (10–50)	Medium (50–150)	        Detects more edges (noisy)
# Medium (50)	High (150)	            Balanced
# High (100)	Very High (200–300)     	Fewer but clean edges
#cv2.Canny(image, threshold1, threshold2)
edges1= cv2.Canny(blur, 50,150)
edges2= cv2.Canny(blur, 100,200)


cv2.imshow("Original", edges2)
cv2.imshow("Blur", edges1)
cv2.waitKey(0)
cv2.destroyAllWindows()
