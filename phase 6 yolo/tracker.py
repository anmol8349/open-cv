from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)


prev_centers ={}

next_id=0



LINE_Y = 300


count = 0
passed_ids = set()

last_y = {}

while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    
    
    results = model(frame , conf=0.4, classes=[0] , verbose = False)
    
    r = results[0]
    
    person_centers=[]
    
    if r.boxes is not None:
        
        for box , cls_id in zip(r.boxes.xyxy , r.boxes.cls):
            cls_id = int(cls_id)
            
            if cls_id == 0:
                x1,y1,x2,y2 = map(int, box)
                cx = int ((x1+x2)/2)
                cy = int ((y1+y2)/2)
                
                person_centers.append((cx , cy))
                
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
    new_centers ={}
    
    for (cx, cy) in person_centers:
        best_id = None
        best_dist = 50
        
        for pid, (px, py) in prev_centers.items():
            
            dist = ((cx - px)**2 + (cy - py)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = pid
                
                
        if best_id is None:
            pid = next_id
            next_id += 1
        else:
            pid = best_id
            
            
        new_centers[pid] = (cx, cy)
        
        if cy > LINE_Y and pid not in passed_ids:
            passed_ids.add(pid)
            count += 1

            
    prev_centers = new_centers
    
     # LINE CROSSING COUNT (TOP -> BOTTOM)
    for pid, (cx, cy) in prev_centers.items():

        if pid in last_y:
            if last_y[pid] < LINE_Y and cy >= LINE_Y:
                cross_count += 1

        last_y[pid] = cy

    
    cv2.putText(frame,
            f"Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,0,0), 2)
    
    
    frame = r.plot()
    
    for pid, (cx, cy) in prev_centers.items():
        cv2.putText(frame, str(pid), (cx+5, cy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                
    for (cx, cy) in person_centers:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
    print("Person centers:", person_centers) 
    
    
    cv2.line(frame,
         (0, LINE_Y),
         (frame.shape[1], LINE_Y),
         (0,0,255), 2)

    
    cv2.imshow("PERSON only + centers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()       