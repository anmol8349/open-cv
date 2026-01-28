# A contour = outline of a shape.
'''A contour is a continuous curve (boundary) joining all 
points along the boundary of an object that have the same color or
intensity.'''

import cv2
import numpy as np


def tobgr(imz):
    if len(imz.shape)==2:
        return cv2.cvtColor(imz , cv2.COLOR_GRAY2BGR)
    return imz




imgs = cv2.imread("shape.png")

img=cv2.resize(imgs,(725,450))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
edges = cv2.Canny(blur, 50, 70)


_, thresh = cv2.threshold(blur,233,255,cv2.THRESH_BINARY)


contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#DRAW CONTOUR
imgzt=cv2.drawContours(img,contours,-1,(56,50,255),2)



img = tobgr(imgzt)
gray=tobgr(gray)
blur = tobgr(blur)
edges= tobgr(edges)
thresh = tobgr(thresh)


print("total contour: ",len(contours))

top = np.hstack((imgzt,gray))
bottom = np.hstack((blur,thresh))
final = np.vstack((top,bottom))
cv2.imshow("stack",final)






cv2.waitKey(0)
cv2.destroyAllWindow()