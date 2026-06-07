from ultralytics import YOLO
import cv2
import cvzone

model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture("person.mp4") 

while True:
    
    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame , conf=0.4 , verbose = False)
    
    r = results[0]
    
    person_centers=[]
    
    if r.boxes is not None:
        
        for box , cls_id in zip(r.boxes.xyxy , r.boxes.cls):
            cls_id = int(cls_id)
            
            if cls_id == 0:
                x1,y1,x2,y2 = map(int, box)
                
                w,h = x2-x1,y2-y1
                
                cvzone.cornerRect(frame,(x1,y1,w,h))
                cx = int ((x1+x2)/2)
                cy = int ((y1+y2)/2)
                
                person_centers.append((cx , cy))

                
    for (cx, cy) in person_centers:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
    # print("Person centers:", person_centers) 
    
    cv2.imshow("PERSON only + centers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()       