import cv2
import numpy as np

# ===============================
# 1. VIDEO INPUT
# ===============================

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# ===============================
# 2. BACKGROUND SUBTRACTOR
# ===============================

# This object learns what the background looks like
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # how many frames to learn background
    varThreshold=50,    # sensitivity (lower = more motion)
    detectShadows=True  # shadows will be gray (127)
)

# ===============================
# 3. MORPHOLOGY KERNEL
# ===============================

# Small brush used to clean noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ===============================
# 4. ROI & LINE SETTINGS
# ===============================

# Region Of Interest (only analyze this area)
ROI_X1, ROI_Y1 = 100, 200
ROI_X2, ROI_Y2 = 600, 450

# Horizontal line for analytics (counting)
LINE_Y = 330

# Store objects that already crossed the line
crossed_ids = set()

# ===============================
# 5. MAIN LOOP
# ===============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===============================
    # 6. DRAW ROI & LINE (VISUALIZATION)
    # ===============================

    # Draw ROI rectangle
    cv2.rectangle(frame,
                  (ROI_X1, ROI_Y1),
                  (ROI_X2, ROI_Y2),
                  (255, 0, 0), 2)

    # Draw counting line
    cv2.line(frame,
             (ROI_X1, LINE_Y),
             (ROI_X2, LINE_Y),
             (0, 0, 255), 2)

    # ===============================
    # 7. CROP ROI
    # ===============================

    # Only process this part of the frame
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # ===============================
    # 8. APPLY BACKGROUND SUBTRACTOR
    # ===============================

    # Foreground mask: white = moving object
    fg_mask = bg_subtractor.apply(roi)

    # ===============================
    # 9. REMOVE SHADOWS
    # ===============================

    # Shadows are gray (127), objects are white (255)
    _, fg_mask = cv2.threshold(
        fg_mask, 200, 255, cv2.THRESH_BINARY
    )

    # ===============================
    # 10. MORPHOLOGY CLEANING
    # ===============================

    # OPENING: remove small white noise
    fg_mask = cv2.morphologyEx(
        fg_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    # CLOSING: fill holes inside objects
    fg_mask = cv2.morphologyEx(
        fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # ===============================
    # 11. FIND MOVING OBJECTS
    # ===============================

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ===============================
    # 12. LOOP THROUGH OBJECTS
    # ===============================

    for idx, cnt in enumerate(contours):

        # Ignore very small areas (noise)
        if cv2.contourArea(cnt) < 1500:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Convert ROI coordinates → frame coordinates
        fx = x + ROI_X1
        fy = y + ROI_Y1

        # Draw bounding box
        cv2.rectangle(frame,
                      (fx, fy),
                      (fx + w, fy + h),
                      (0, 255, 0), 2)

        # ===============================
        # 13. CENTROID (BASIC TRACKING)
        # ===============================

        cx = fx + w // 2
        cy = fy + h // 2

        # Draw centroid
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # ===============================
        # 14. LINE CROSSING LOGIC
        # ===============================

        if cy > LINE_Y and idx not in crossed_ids:
            crossed_ids.add(idx)

    # ===============================
    # 15. DISPLAY COUNT
    # ===============================

    cv2.putText(frame,
                f"Count: {len(crossed_ids)}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # ===============================
    # 16. SHOW WINDOWS
    # ===============================

    cv2.imshow("Analytics Core", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 17. CLEANUP
# ===============================

cap.release()
cv2.destroyAllWindows()
