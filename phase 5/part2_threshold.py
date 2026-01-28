# part2_threshold.py
import cv2

cap = cv2.VideoCapture(0)

bg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret , frame = cap.read()
    
    if not ret:
        break
    
    fg =bg.apply(frame)
    
    _ , fg_clean = cv2.threshold(fg,200,255,cv2.THRESH_BINARY)
    
    
    cv2.imshow("Raw Mask", fg)
    cv2.imshow("Clean Mask", fg_clean)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()