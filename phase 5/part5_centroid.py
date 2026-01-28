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
    _, fg = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x,y,w,h = cv2.boundingRect(cnt)
            cx = x + w//2
            cy = y + h//2

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

    cv2.imshow("Centroid Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
