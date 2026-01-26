import cv2

img = cv2.imread('rj.jpg')
print(img.shape)
# resized = cv2.resize(img,(900,400))  # W,H


resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# ------------------------------------------------------------
h, w = img.shape[:2]

new_width = 800
scale = new_width / w
new_height = int(h * scale)

resiz = cv2.resize(img, (new_width, new_height))


# //////////////////////------------------------------


# img[y1:y2, x1:x2]

cropped = img[230:600, 520:1000]



# ---------------------------------



centre = (w//2,h//2)


angle = 90

scale= 1.0

M = cv2.getRotationMatrix2D(centre,angle , scale)


rotated = cv2.warpAffine(img,M,(w, h))


# cv2.imshow("rot", rotated)


# -----------------------------------



flip_h = cv2.flip(img, 1)   # horizontal
flip_v = cv2.flip(img, 0)   # vertical

cv2.imshow("Horizontal Flip", flip_h)
cv2.imshow("Vertical Flip", flip_v)









# cv2.imshow("original",img)

# cv2.imshow("resized", resiz)

# cv2.imshow("cropped", cropped)

cv2.waitKey(0)
cv2.destroyAllWindow()