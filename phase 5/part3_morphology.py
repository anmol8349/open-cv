import cv2
import numpy as np


cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2(500, 50, True)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fg = bg.apply(frame)
    _, fg = cv2.threshold(fg, 190, 255, cv2.THRESH_BINARY)
    
    fg_open = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_close = cv2.morphologyEx(fg_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow("Before Morphology", fg)
    cv2.imshow("After Morphology", fg_close)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
