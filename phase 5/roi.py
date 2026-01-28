# ROI = process only a part of the frame.
'''✔ Performance boost
✔ Focus detection area
✔ Avoid background noise
✔ Surveillance zones  '''


import cv2

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

ROI_X1, ROI_Y1 = 200, 100
ROI_X2, ROI_Y2 = 600, 400

while cap.isOpened():
    roi1 = frame1[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    roi2 = frame2[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    diff = cv2.absdiff(roi1, roi2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x,y,w,h = cv2.boundingRect(cnt)

            # Convert ROI coords → frame coords
            cv2.rectangle(frame1,
                          (x+ROI_X1, y+ROI_Y1),
                          (x+ROI_X1+w, y+ROI_Y1+h),
                          (0,255,0), 2)

    cv2.rectangle(frame1,
                  (ROI_X1, ROI_Y1),
                  (ROI_X2, ROI_Y2),
                  (255,0,0), 2)

    cv2.imshow("Motion + ROI", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
