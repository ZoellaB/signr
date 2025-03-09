import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

cap = cv2.VideoCapture(1)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detect(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

# #0: Face
# #1: Pose
# #2: Left hand
# #3: Right hand
# def thinning_landmarks(img, results, num):
#     match num:
#         case 0:
#             mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
#             return
#         case 1:
#             mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
#             return
#         case 2:
#             mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
#             return
#         case 3:
#             mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
#             return
#         case _:
#             return

def draw_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        img, results = mediapipe_detect(frame, holistic)
        draw_landmarks(img, results)
        # thinning_landmarks(img, results, 0)
        # thinning_landmarks(img, results, 3)

        

        cv2.imshow('Cam Feed', img)
        if (cv2.waitKey(30) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            cap.release()
            break
