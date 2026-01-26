# cv2.imread() → reads file into memory

# If path wrong → it returns None

# cv2.imshow() → opens a window

# cv2.waitKey(0) → waits until key press

# cv2.destroyAllWindows() → closes window



import cv2

img=cv2.imread("rj.jpg")

# if img is not None:
#     cv2.imshow("Window title",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
# else:
#     print("Error : image not found ")




print("Shape:", img.shape)   # (height, width, channels)
print("Size:", img.size)     # total number of values
print("Type:", img.dtype)    # usually  uint8



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Window title",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

