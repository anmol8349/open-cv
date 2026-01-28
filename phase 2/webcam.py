import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ff=cv2.flip(frame,0)

    cv2.imshow("Webcam Feed", ff)
    cv2.imshow("Grayscale Feed", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
