from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture("person.mp4") 


tracks = {}
next_id = 0

# --------------------------------------

passed_ids = set()
count = 0

# ---------------------------------------------------------
line_y = 300

while True:
    
    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame , conf=0.4 , verbose = False)
    
    r = results[0]
    
    persons=[]
    
    if r.boxes is not None:
        
        for box , cls_id in zip(r.boxes.xyxy , r.boxes.cls):
            cls_id = int(cls_id)
            
            if cls_id == 0:
                x1,y1,x2,y2 = map(int, box)
                
                w,h = x2-x1,y2-y1
                
                cvzone.cornerRect(frame,(x1,y1,w,h))
                
                cx = int ((x1+x2)/2)
                cy = int ((y1+y2)/2)
                
                persons.append((cx, cy, x1, y1, x2, y2))


    new_tracks = {}
    user_id=set()
    for cx, cy, x1, y1, x2, y2 in persons:
        matched_id = None
        min_dist = 1e9
        
        for tid, (px, py) in tracks.items():
            
            if tid in user_id:
                continue

            dist = math.hypot(cx - px, cy - py)

            if dist < 50 and dist < min_dist:
                min_dist = dist
                matched_id = tid

        # assign ID
        if matched_id is None:
            track_id = next_id
            next_id += 1
            prev_cy = cy   # first time, no previous info
        else:
            track_id = matched_id
            prev_cy = tracks[track_id][1]
            
            
        user_id.add(track_id)
        new_tracks[track_id] = (cx, cy)
        
        # STEP 3 : real line crossing logic
        # -----------------------------------
        # crossing from top -> bottom
        if prev_cy < line_y and cy >= line_y:
            if track_id not in passed_ids:
                passed_ids.add(track_id)
                count += 1
        # -----------------------------------
        # Draw person box and center
        # -----------------------------------
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        cv2.putText(frame,f"ID {track_id}",
                    (x1, y1 - 8),cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    # update tracker
    tracks = new_tracks
    
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)
    
    cv2.putText(frame,
                f"Count : {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1,
                (0,0,255),
                2)   
                 
    # for (cx, cy) in person_centers:
    #     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
    cv2.imshow("PERSON only + centers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()       