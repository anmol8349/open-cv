# import cv2

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7, 7), 0)
#     edges = cv2.Canny(blur, 50, 150)
    
#     cv2.imshow("Original", frame)
#     cv2.imshow("Gray", gray)
#     cv2.imshow("Edges", edges)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

# Create a control window
cv2.namedWindow("Controls")

# Trackbars for Canny + Blur
cv2.createTrackbar("T1", "Controls", 50, 500, nothing)
cv2.createTrackbar("T2", "Controls", 150, 500, nothing)
cv2.createTrackbar("Blur", "Controls", 7, 31, nothing)  # must be odd

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Read trackbar values
    t1 = cv2.getTrackbarPos("T1", "Controls")
    t2 = cv2.getTrackbarPos("T2", "Controls")
    k  = cv2.getTrackbarPos("Blur", "Controls")

    # Ensure valid values
    if k % 2 == 0:
        k += 1
    if k < 1:
        k = 1
    if t1 > t2:
        t1, t2 = t2, t1

    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blur, t1, t2)
    
    cv2.imshow("Original", frame)
    cv2.imshow("Gray", gray)
    cv2.imshow("Edges", edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
