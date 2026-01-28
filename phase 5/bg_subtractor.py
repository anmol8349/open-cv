import cv2

cap=cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv2.imshow("frame", frame)
    cv2.imshow("fore Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
