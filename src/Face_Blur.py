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
    if landmarks.size == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    convexhull = cv2.convexHull(landmarks)

    # 2. Face blurring
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, convexhull, 255)

    # Extract the face
    frame_copy = cv2.blur(frame_copy, (27, 27))
    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

    # Extract background
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    # Final result
    result = cv2.add(background, face_extracted)

    cv2.imshow("Frame", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
