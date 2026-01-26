import cv2

def nothing(x):
    pass

img = cv2.imread("rj.jpg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

cv2.namedWindow("Edges")
# cv2.createTrackbar(trackbar_name, window_name, start_value, max_value, callback)

cv2.createTrackbar("T1", "Edges", 50, 500, nothing)
cv2.createTrackbar("T2", "Edges", 150, 500, nothing)

while True:
    t1 = cv2.getTrackbarPos("T1", "Edges")
    t2 = cv2.getTrackbarPos("T2", "Edges")

    edgesin = cv2.Canny(blur, t1, t2)

    cv2.imshow("Edges", edgesin)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
