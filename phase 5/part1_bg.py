import cv2

cap = cv2.VideoCapture(0)

bg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg.apply(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
