from ultralytics import YOLO
import cv2
import math
import cvzone



model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("person.mp4") 

tracks ={}
next_id = 0

line_x = 500

count = 0

passed_ids = set()


while True:

    ret, frame = cap.read()

    if not ret:
        break
    
   
    # cv2.imshow("Frame", frame)
    results = model(frame , conf=0.4 , verbose = False)
    persons=[]
    r = results[0]
    
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
                
                
                cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )
                
        new_tracks = {}
        used_ids = set()
            
            
        for cx, cy, x1, y1, x2, y2 in persons:

            matched_id = None
            min_dist = 1e9
            
            for tid, (px, py) in tracks.items():

                if tid in used_ids:
                    continue

                dist = math.hypot(
                    cx - px,
                    cy - py
                )

                if dist < 50 and dist < min_dist:

                    min_dist = dist
                    matched_id = tid
                    
            if matched_id is None:

                track_id = next_id
                next_id += 1

            else:

                track_id = matched_id
                
            used_ids.add(track_id)
            prev = tracks.get(track_id)
            
            if prev is not None:

                prev_cx, prev_cy = prev

                if prev_cx > line_x and cx <= line_x:

                    if track_id not in passed_ids:

                        passed_ids.add(track_id)

                        count += 1
                        
                        
            new_tracks[track_id] = (cx, cy)

            cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )
        
        tracks = new_tracks    

                
    cv2.line(
            frame,
            (line_x, 0),
            (line_x,frame.shape[0]),
            (255, 0, 0),
            2
        )
    
    cv2.putText(
            frame,
            f"Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

