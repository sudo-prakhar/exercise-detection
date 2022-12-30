import pickle
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = 'models/knn.sav'

model = pickle.load(open(MODEL_PATH, 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    pose_landmarks = results.pose_landmarks

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    image = cv2.flip(image, 1)

    if pose_landmarks is not None:
      pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
      frame_height, frame_width = image.shape[:2]
      pose_landmarks *= np.array([frame_width, frame_height, frame_width])
      pose_landmarks = np.around(pose_landmarks, 5).flatten()
      pose_landmarks = pose_landmarks.reshape(1, -1)

      prediction = model.predict(pose_landmarks)

      cv2.putText(img=image, text=str(prediction), org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 0, 0),thickness=3)

  
  
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()