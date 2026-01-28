import cv2
import numpy as np


imgs = cv2.imread("shape.png")

img=cv2.resize(imgs,(825,600))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
edges = cv2.Canny(blur, 50, 70)


_, thresh = cv2.threshold(blur,233,255,cv2.THRESH_BINARY_INV)


contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


print("total contour: ",len(contours))

for i, cnt in enumerate(contours):

    print(f"\nContour #{i+1}")


    area = cv2.contourArea(cnt)
    print("Area:", area)
    
    
    perimeter = cv2.arcLength(cnt, True)
    print("Perimeter:", perimeter)
    
    
    
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print("Vertices:", len(approx))

    cv2.drawContours(img,[cnt],-1,(56,50,255),2)
    
    
    
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    
     # ---- 4.4 Shape Detection ----
    shape = "Unknown"

    if len(approx) == 3:
        shape = "Triangle"

    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.95 < aspect_ratio < 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif len(approx) == 5:
        shape = "Pentagon"

    else:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 0.90 < aspect_ratio < 1.10:
            shape = "Circle"
        else:
            shape = "Oval"


    print("Detected Shape:", shape)


    cv2.putText(
        img,
        f"Area: {int(area)}",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )
    
    
    cv2.putText(
        img,
        shape,
        (x+10, y +20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1
    )
    
    
    
    
# cv2.imshow("stack",thresh)

cv2.imshow("stack",img)

cv2.waitKey(0)
cv2.destroyAllWindow()