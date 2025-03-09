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
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=3))
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  mp_drawing.DrawingSpec(color=(128,0,128), thickness=5, circle_radius=5), mp_drawing.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3))
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def flatten(results):

    # print(results.pose_landmarks.landmark)
    # print(len(results.pose_landmarks.landmark))
    # print(results.left_hand_landmarks.landmark)
    # print(len(results.left_hand_landmarks.landmark))
    # print(results.right_hand_landmarks.landmark)
    # print(len(results.right_hand_landmarks.landmark))


    # pose_key_pt_vals = np.array([[res.x, res.y,res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    # if(results.left_hand_landmarks):
    #     lefthand_key_pt_vals = np.array([[res.x, res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    # else: 
    #     lefthand_key_pt_vals = np.zeros(21*3)
    # if(results.right_hand_landmarks):
    #     righthand_key_pt_vals = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    # else:
    #     righthand_key_pt_vals = np.zeros(21*3)

    print(type(results.left_hand_landmarks))
    print(type(results.left_hand_landmarks.landmark))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # print(pose)
    # print(len(pose))
    # print(face)
    # print(len(face))
    # print(lh)
    # print(len(lh))
    # print(rh)
    # print(len(rh))



with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        img, results = mediapipe_detect(frame, holistic)
        draw_landmarks(img, results)
        
        cv2.imshow('Cam Feed', img)

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        if (cv2.waitKey(30) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            cap.release()
            break


    flatten(results)


