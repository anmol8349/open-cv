import cv2


cap = cv2.VideoCapture("vid.mp4")

if not cap.isOpened():
    print("video not loaded")
    exit()


while True:
    
    ret,frame = cap.read()
    
    if not ret:
        print("Video ended.")
        break
    
    
    h, w = frame.shape[:2]
    
    max_height = 700
    scale = max_height / h
    nw = int(w*scale)
    nh = int(h*scale)
    
    
    frame = cv2.resize(frame, (nw,nh))

    # if h > w:  # portrait detected
    #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
 
 
 
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    cv2.imshow('Video player' , frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    
    
    # 0xFF==27 for= esc
    # != -1         ===any key
    
        
cap.release()
cv2.destroyAllWindows()