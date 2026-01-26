import cv2

img = cv2.imread('rj.jpg')
h,w = img.shape[:2]


x=(520,1000)
y=(230,600)
color= (0, 100, 255)
# cv2.line(image, start_point, end_point, color, thickness)



line= cv2.line(img, (0,0), (100,500),(0, 255, 0) , 4)


# cv2.rectangle(image, top_left, bottom_right, color, thickness)

cv2.rectangle(img,y,x,color,5)



# cv2.putText(image, text, org, font, fontScale, color, thickness)
cv2.putText(img, "Anmol", (800, 200),cv2.FONT_HERSHEY_SIMPLEX,1.2 , color, 5)



cv2.circle(img, (300, 200), 50, (255, 0, 0), 5)





cv2.imshow("main",line)
# cv2.imshow("mains",img)



cv2.imwrite("edited.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindow()