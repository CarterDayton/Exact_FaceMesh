import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# Load face landmarks
fl = FaceLandmarks()

cap = cv2.VideoCapture("stockVid.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # 1. Face landmarks detection
    landmarks = fl.get_facial_landmarks(frame)
    convexhull = cv2.convexHull(landmarks)

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(frame, [convexhull], True, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
