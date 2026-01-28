# Thresholding converts a grayscale image into a black & white (binary) image.

# If pixel value > THRESHOLD → make it WHITE (255)
# If pixel value ≤ THRESHOLD → make it BLACK (0)


# Object detection ,OCR (text reading) , Shape detection ,Medical imaging ,Barcode scanning ,Background removal


import cv2

def nothing(x):
    pass

img = cv2.imread("shape.png")

if img is None:
    print("Image not found")
    exit()
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("thresholded tool")

cv2.createTrackbar("T1","thresholded tool",10,255,nothing)
cv2.createTrackbar("T2","thresholded tool",10,255,nothing)


while True:
    
    t1 =cv2.getTrackbarPos("T1" ,"thresholded tool")
    t2 = cv2.getTrackbarPos("T2","thresholded tool")
    
    _, thresh = cv2.threshold(gray, t1, t2, cv2.THRESH_BINARY)

    # cv2.imshow("Original", gray)
    cv2.imshow("Thresholded tool", thresh)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()

