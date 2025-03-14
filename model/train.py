import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json
import sys
from multiprocessing import Process
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

def get_keypts_orchestrator(start=0):
    dir = "training_data"
    KEYPTS_PATH = os.path.join(os.curdir, 'gen_keypoints_data')
    if not os.path.exists(KEYPTS_PATH):
        os.makedirs(KEYPTS_PATH)
    word_count=0


    #Save missing video instances
    missing = open(dir + "/missing.txt").readlines()

    #Save word classification from 0-1999 as dict
    word_class = {}
    with open(dir + "/wlasl_class_list.txt") as class_file:
        for line in class_file:
            if (len(line.split(maxsplit=1)) != 2):
                sys.exit("Error using wlasl class list")
            val, key = line.split(maxsplit=1) #class number is val, and name of word is key
            word_class[key.strip()] = str(val)

    # Open the JSON file
    with open(dir + '/WLASL_v0.3.json') as f:
        # Load the JSON data
        json_data = json.load(f)
    # Check if the data is a list of dictionaries
    if isinstance(json_data, list):
        # Convert the JSON array into individual dictionaries
        train_list = [dict(item) for item in json_data if isinstance(item, dict)]
        
    else: # may break here.
        print("The JSON file does not contain an array.")
    # Trains per word, go to each 1+ instances
    for word in train_list:
        if(word_count < start):
            word_count += 1
            continue
        word_count += 1
        print("Training " + word['gloss'])
        for instance in word['instances']:
            try: 
                os.makedirs(os.path.join(KEYPTS_PATH, word_class[str(word['gloss'])], str(instance['video_id'])))
            except:
                pass
            if instance['video_id'] not in missing:
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

                        #Showing, not neccesary when training
                        draw_landmarks(img, results)
                        cv2.imshow('Cam Feed', img)

                        #Saving keypoints:
                        keypts = get_keypts(results)
                        numpy_path = os.path.join(KEYPTS_PATH, word_class[str(word['gloss'])], str(instance['video_id']),str(frame_count))
                        if not os.path.exists(numpy_path):
                            np.save(numpy_path, keypts)
       
                        if (cv2.waitKey(1) & 0xFF == ord('q')):
                            continue
                    cap.release()
    
    
    
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    p = Process(target=get_keypts_orchestrator) #stopped at 411 309 68 87 157 180 689 122 1228
    p.start()
    p.join()



# class_word = {}
# with open("training_data" + "/wlasl_class_list.txt") as class_file:
#     for line in class_file:
#         if (len(line.split(maxsplit=1)) != 2):
#             sys.exit("Error using wlasl class list")
#         key, val = line.split(maxsplit=1) #class number is val, and name of word is key
#         class_word[key.strip()] = str(val)

# print(class_word)


    
# sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
# np.array(sequences).shape
# np.array(labels).shape
# X = np.array(sequences)


# X.shape
# y = to_categorical(labels).astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# y_test.shape

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
# res = [.7, 0.2, 0.1]
# actions[np.argmax(res)]
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
# model.summary()