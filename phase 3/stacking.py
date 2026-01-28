import cv2
import numpy as np

imgs = cv2.imread("rj.jpg")
img = cv2.resize(imgs, None, fx=0.4, fy=0.5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
edges = cv2.Canny(blur, 50, 150)

# Convert gray & edges to 3-channel (important!)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Stack horizontally
top_row = np.hstack((img, gray_bgr))
bottom_row = np.hstack((blur_bgr := cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), edges_bgr))

# Stack vertically
final_stack = np.vstack((top_row, bottom_row))




cv2.imshow("Stacked Output", final_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def to_bgr(img):
#     if len(img.shape) == 2:
#         return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img

# img = cv2.imread("resources/lena.png")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 0)
# edges = cv2.Canny(blur, 50, 150)

# img = to_bgr(img)
# gray = to_bgr(gray)
# blur = to_bgr(blur)
# edges = to_bgr(edges)

# top = np.hstack((img, gray))
# bottom = np.hstack((blur, edges))
# final = np.vstack((top, bottom))

# cv2.imshow("Stacked", final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
