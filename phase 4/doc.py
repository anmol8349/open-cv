import cv2
import numpy as np

# -------------------------------
# Helper Functions
# -------------------------------

def reorder(points):
    """
    Reorder 4 corner points in consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    new_points[0] = points[np.argmin(s)]   # top-left
    new_points[2] = points[np.argmax(s)]   # bottom-right

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # top-right
    new_points[3] = points[np.argmax(diff)]  # bottom-left

    return new_points


def biggest_contour(contours):
    """
    Find the biggest 4-point contour (document)
    """
    biggest = np.array([])
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:   # ignore tiny noise
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def preprocess(img):
    """
    Preprocessing pipeline for edge detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)
    return gray, blur, edges


# -------------------------------
# Main Program
# -------------------------------

img = cv2.imread("doc.jpg")

if img is None:
    print(" Image not found. Check file path.")
    exit()

height, width = img.shape[:2]

# 1  Preprocess image
gray, blur, edges = preprocess(img)

# 2  Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# 3  Find biggest 4-point contour
biggest, max_area = biggest_contour(contours)

if biggest.size == 0:
    print("No document detected.")
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# 4 Draw detected document
img_contour = img.copy()
cv2.drawContours(img_contour, [biggest], -1, (0, 255, 0), 3)

# 5  Reorder corner points
points = reorder(biggest)

# 6️ Define target points (output size)
output_width = 600
output_height = 800

pts1 = np.float32(points)
pts2 = np.float32([
    [0, 0],
    [output_width, 0],
    [output_width, output_height],
    [0, output_height]
])

# 7  Perspective transform
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warp = cv2.warpPerspective(img, matrix,
                           (output_width, output_height))

# 8  Post-processing (optional)
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
warp_thresh = cv2.adaptiveThreshold(
    warp_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# -------------------------------
# Display Results
# -------------------------------

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Document", img_contour)
cv2.imshow("Scanned Output", warp)
cv2.imshow("Scanned (Black & White)", warp_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
