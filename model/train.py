import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json
import sys
from multiprocessing import Pool
# from sklearn.model_selection import train_test_split
    
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
TRAIN_DIR = "training_data"
GEN_DIR = 'gen_keypoints_data'
START = 0
KEYPTS_PATH = os.path.join(os.curdir, GEN_DIR)

word_class = None
missing = None

def mediapipe_detect(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

def draw_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2))
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  mp_drawing.DrawingSpec(color=(128,0,128), thickness=1, circle_radius=2), mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2))
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

def get_word_classification(dir):
    #Save word classification from 0-1999 as dict
    word_class = {}
    with open(TRAIN_DIR + dir) as class_file:
        for line in class_file:
            if (len(line.split(maxsplit=1)) != 2):
                sys.exit("Error using wlasl class list")
            val, key = line.split(maxsplit=1) #class number is val, and name of word is key
            word_class[key.strip()] = str(val)
    return word_class

def get_missing_vids(miss_dir):
    miss = []
    miss = open(TRAIN_DIR + miss_dir).readlines()  #Save missing video instances
    return miss

def get_vid_metadata(dir):
    train_list = []
    
    # Open the JSON file
    with open(TRAIN_DIR + dir) as f:
        # Load the JSON data
        json_data = json.load(f)
    # Check if the data is a list of dictionaries
    if isinstance(json_data, list):
        # Convert the JSON array into individual dictionaries
        train_list = [dict(item) for item in json_data if isinstance(item, dict)]
        
    else: # may break here.
        print("The JSON file does not contain an array.")

    return train_list

def get_keypts_orchestrator(word):
    # Trains per word, go to each 1+ instances
    print("Training " + word['gloss'])
    for instance in word['instances']:
        try: 
            os.makedirs(os.path.join(KEYPTS_PATH, word_class[str(word['gloss'])]))
        except:
            pass

        if instance['video_id'] not in missing:
            os.makedirs(os.path.join(KEYPTS_PATH, word_class[str(word['gloss'])], str(instance['video_id'])))
            with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
                cap = cv2.VideoCapture('training_data/videos/' + instance['video_id'] +  '.mp4')
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Ignoring empty camera frame.")
                        break 
                    img, results = mediapipe_detect(frame, holistic)
                    frame_count += 1

                    # #Showing, not neccesary when training
                    # draw_landmarks(img, results)
                    # cv2.imshow('Cam Feed', img)

                    #Saving keypoints:
                    keypts = get_keypts(results)
                    numpy_path = os.path.join(KEYPTS_PATH, word_class[str(word['gloss'])], str(instance['video_id']),str(frame_count))
                    if not os.path.exists(numpy_path):
                        np.save(numpy_path, keypts)
    
                    if (cv2.waitKey(1) & 0xFF == ord('q')):
                        continue
                cap.release()
    cv2.destroyAllWindows()


def init_worker(wc_dir, missword_dir):
    global word_class, missing
    word_class = get_word_classification(wc_dir)
    missing = get_missing_vids(missword_dir)

def main():
    global word_class, missing

    if not os.path.exists(KEYPTS_PATH):
        os.makedirs(KEYPTS_PATH)

    train_list = get_vid_metadata("/WLASL_v0.3.json")
    with Pool(initializer=init_worker, initargs=("/wlasl_class_list.txt", "/missing.txt")) as pool:
        pool.imap_unordered(get_keypts_orchestrator,train_list)
        pool.close()
        pool.join()



if __name__ == '__main__':
    main()