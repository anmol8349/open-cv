from ultralytics import YOLO
import cv2
import math
import cvzone
# ============================================
        # TRACK PEOPLE 
# ============================================

def track_people(persons, tracks, next_id):

    new_tracks = {}
    used_ids = set()

    tracked_persons = []

    for cx, cy, x1, y1, x2, y2 in persons:

        matched_id = None
        min_dist = 1e9

        for tid, track in tracks.items():
            
            if tid in used_ids:
                continue
            
            
            predicted_x = track["x"] + track["vx"]
            predicted_y = track["y"] + track["vy"]
            


            dist = math.hypot(
                cx - predicted_x,
                cy - predicted_y
            )

            if dist < 100 and dist < min_dist:

                min_dist = dist
                matched_id = tid
        
        
        if matched_id is None:
            
            track_id = next_id
            next_id+=1

            vx = 0
            vy = 0 
        else:
            track_id = matched_id
            
            old_track = tracks[matched_id]
            
            predicted_x = old_track["x"] + old_track["vx"]
            predicted_y = old_track["y"] + old_track["vy"]
            
            vx = cx - old_track["x"]
            vy = cy - old_track["y"]
        

        used_ids.add(track_id)


        if matched_id is None:
            confidence = 1.0
        else:
            confidence = min(old_track["confidence"] + 0.1, 1.0 )
        
        new_tracks[track_id] = {
                              "x":cx,
                              "y":cy,
                              "vx":vx,
                              "vy":vy,
                              "age":0,
                              "confidence":1.0
                                 }

        tracked_persons.append(
            (
                track_id,
                cx, cy,
                x1, y1, x2, y2
            )
        )
        
    for tid, track in tracks.items():

        if tid not in used_ids:

            track["age"] += 1
            
            track["confidence"] -= 0.1
            
            
            if track["age"] <= 5 and track["confidence"] > 0 :   

                new_tracks[tid] = track



    return tracked_persons, new_tracks, next_id


# ============================================
           # DIRECTION
# ============================================

def get_direction(track_id, cy, prev_centers):

    direction = "Unknown"

    if track_id in prev_centers:

        prev_cx, prev_cy = prev_centers[track_id]

        if cy < prev_cy:
            direction = "UP"

        elif cy > prev_cy:
            direction = "DOWN"

    return direction



# =========================================
        #  UI DRAWING 
# =========================================

def draw_dashboard(
    frame,
    entry_count,
    exit_count,
    line_y,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2
):

    cv2.line(
        frame,
        (0,line_y),
        (frame.shape[1],line_y ),
        (255, 0, 0),
        2
    )

    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255,250,0),
        2
    )

    cv2.putText(
        frame,
        f"Entry: {entry_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Exit: {exit_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


# =========================================
        #  predict 
# =========================================


def predict_next_position(track):
    
    predicted_x = track["x"] + track["vx"]
    predicted_y = track["y"] + track["vy"]

    return predicted_x , predicted_y



# =========================================
        #  detection 
# =========================================


def detect_people(frame,model):
    
    persons =[]
    
    results= model(frame,conf=0.4, verbose=False)
    
    r = results[0]
    
    if r.boxes is not None:
    
        for box , cls_id in zip(r.boxes.xyxy , r.boxes.cls):
            cls_id = int(cls_id)
            
            if cls_id == 0:
                x1,y1,x2,y2 = map(int, box)
                
                w,h = x2-x1,y2-y1
                
                cvzone.cornerRect(frame,(x1,y1,w,h),l=15,t=2,colorC=(150, 250, 0))
                
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
    
    return persons




model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos\\person.mp4") 


tracks ={}
next_id = 0


track_history ={}

prev_centers = {}


predicted_positions = {}




line_y = 300



entry_count = 0 
exit_count = 0
counted_ids = set()



roi_x1=200
roi_y1 = 100

roi_x2 = 800
roi_y2=800

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
   
    # cv2.imshow("Frame", frame)
    
    persons = detect_people(frame,model)
    
    tracked_persons, new_tracks, next_id = track_people(persons,tracks,next_id)
    


    for track_id,cx,cy,x1,y1,x2 ,y2 in tracked_persons: 
        
        inside_roi = (roi_x1<=cx<=roi_x2 and roi_y1<= cy <= roi_y2)
        
        if not inside_roi:
            continue
        
        
        
        prev = tracks.get(track_id)
        
        if prev is not None:
            prev_cx = prev["x"]
            prev_cy = prev["y"]
  
        
        
        direction = get_direction(track_id , cy ,prev_centers)
           
           
           
                
        if track_id in prev_centers and track_id not in counted_ids:
            prev_cx , prev_cy = prev_centers[track_id]
            
            #  top to bottom 
            
            if prev_cy < line_y and cy >=line_y:
                entry_count+=1
                counted_ids.add(track_id)
                
            #  bottom to top
                
            elif prev_cy > line_y and cy  <= line_y:
                exit_count+=1
                counted_ids.add(track_id)
                
             
             
        # Store Position History
        
        if track_id not in track_history:
            track_history[track_id]=[]
            
            
            
        track_history[track_id].append((cx,cy))
        
        
        # limit trail length 
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)
            
        points = track_history[track_id]
        trail_length = len(points)
        
        for i in range (1,len(points)):
            cv2.line(frame, points[i-1],points[i],(2,255,255),2)

        cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )
        
        cv2.putText(frame, direction, (x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)
        cv2.putText(frame,f"Trail:{trail_length}",(x1,y2-20),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,0),1)
        
        
        
        
        
        
        if track_id in new_tracks:
           
            
            predicted_x, predicted_y = predict_next_position(new_tracks[track_id])
            
            predicted_positions[track_id]= (predicted_x , predicted_y )
            
            
            if track_id in predicted_positions:
                px , py = predicted_positions[track_id]
                
                cv2.circle(frame,(int(px) , int(py)),6, (255,0,255),-1)
                cv2.putText(frame,
                            "Pred",
                            (int(px), int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            2
                        )
                
                cv2.line(
                            frame,
                            (cx, cy),
                            (int(px) , int(py)),
                            (255, 0, 255),
                            2
                        )
                                
        prev_centers [track_id]= (cx,cy)
        
    tracks = new_tracks    
    

    draw_dashboard(
    frame,
    entry_count,
    exit_count,
    line_y,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2
)         



    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

