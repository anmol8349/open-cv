from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # n = nano (fastest)

# Read image
img = cv2.imread("street.jpg")


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break 

  
        # results = model(frame)
        # annotated = results[0].plot()

        # cv2.imshow("YOLOv8 Webcam", annotated)

        # if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        #     break



# Run detection
results = model(img)

# Show result
annotated = results[0].plot()


cv2.imshow("YOLOv8 Detection", annotated)

cv2.waitKey(0)

# cap.release()
cv2.destroyAllWindows()
