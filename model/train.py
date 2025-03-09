import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json


mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detect(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

def draw_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=3))
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  mp_drawing.DrawingSpec(color=(128,0,128), thickness=5, circle_radius=5), mp_drawing.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3))
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Draw landmark annotation on the image.
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        img,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

def get_keypts(results):    
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def train_video_orchestrator(dir):

    missing = open(dir + "/missing.txt").readlines()

    with open(dir + '/WLASL_v0.3.json') as f:
        print(f)
        d = f[0]
        print(d)
        print(d[0])

    
    
    # cap = cv2.VideoCapture('training_data/videos/sample_video.mp4')

# with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         img, results = mediapipe_detect(frame, holistic)
#         draw_landmarks(img, results)
        
#         cv2.imshow('Cam Feed', img)

#         if not ret:
#             print("Ignoring empty camera frame.")
#             # If loading a video, use 'break' instead of 'continue'.
#             break
#         if (cv2.waitKey(30) & 0xFF == ord('q')):
#             cv2.destroyAllWindows()
#             cap.release()
#             break



    
train_video_orchestrator("training_data")

