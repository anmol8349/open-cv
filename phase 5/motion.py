'''
Docstring for phase 5.motion
Motion detection =
Compare current frame with previous frame
Find what changed
Draw boxes around changes
'''

import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

