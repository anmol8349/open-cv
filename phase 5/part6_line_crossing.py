import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2(500, 50, True)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

ROI_X1, ROI_Y1 = 100, 200
ROI_X2, ROI_Y2 = 600, 450
LINE_Y = 320

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    
    fg = bg.apply(roi)
    
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)
    
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x,y,w,h = cv2.boundingRect(cnt)
            cy = y + h//2
            
            if cy > (LINE_Y - ROI_Y1):
                count += 1
                
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            cv2.rectangle(frame,
                          (x+ROI_X1, y+ROI_Y1),
                          (x+ROI_X1+w, y+ROI_Y1+h),
                          (0,255,0), 2)
            
    cv2.rectangle(frame, (ROI_X1,ROI_Y1), (ROI_X2,ROI_Y2), (255,0,0), 2)
    
    cv2.line(frame, (ROI_X1,LINE_Y), (ROI_X2,LINE_Y), (0,0,255), 2)
    
    
    cv2.putText(frame, f"Count: {count}",
                (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Line Crossing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()