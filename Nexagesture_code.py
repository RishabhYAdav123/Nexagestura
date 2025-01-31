import cv2
# Mediapipe library for real-time computer vision tasks, like hand tracking.
import mediapipe as mp  
# Importing hypot function to calculate the distance between two points.
from math import hypot  
# NumPy library for numerical operations and array handling.
import numpy as np  
# Importing ctypes for low-level C-like data manipulation.
from ctypes import cast, POINTER  
# Importing CLSCTX_ALL to specify the context for COM object creation.
from comtypes import CLSCTX_ALL  
# Pycaw for controlling audio settings in Windows.
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  
# Importing MessageToDict for converting Protobuf messages to dictionaries.
from google.protobuf.json_format import MessageToDict  
# Library for controlling screen brightness on various operating systems.
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)

# Setting up Mediapipe's hand detection model with drawing utilities for detecting and visualizing hand landmarks.
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils


# Accessing the system's audio endpoint to control the speaker volume using the Pycaw library.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#setup for volume limit
volumelimit = volume.GetVolumeRange()
volMin = volumelimit[0]
volMax = volumelimit[1]




# Continuously capturing frames, flipping the image, converting it to RGB, and processing it for hand detection.
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # List to store landmark coordinates of the left hand.
    left_lmList=[]
    # List to store landmark coordinates of the right hand.
    right_lmList = []

    # Check if hand landmarks and handedness (left/right hand) are detected.
    if results.multi_hand_landmarks and results.multi_handedness:
        for i in results.multi_handedness:
            # Get the hand label (Left/Right).
            label = MessageToDict(i)['classification'][0]['label']

            # if left hand is detected
            if label == 'Left':
                for lm in results.multi_hand_landmarks[0].landmark:
                    h, w, _ = img.shape
                    left_lmList.append([int(lm.x * w), int(lm.y * h)])
                # Draw left hand landmarks.
                mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)

            # if the right hand id is detected
            if label == 'Right':
                index = 0
                if len(results.multi_hand_landmarks) == 2:
                    index = 1
                for lm in results.multi_hand_landmarks[index].landmark:
                    h, w, _ = img.shape
                    right_lmList.append([int(lm.x * w), int(lm.y * h)])
                    # Draw left hand landmarks.
                    mpDraw.draw_landmarks(img, results.multi_hand_landmarks[index], mpHands.HAND_CONNECTIONS)

    # if left hand is detected
    if left_lmList != []:
        # Get the coordinates of the thumb tip (index 4) and index finger tip (index 8) from the left hand landmarks.
        x1, y1 = left_lmList[4][0], left_lmList[4][1]
        x2, y2 = left_lmList[8][0], left_lmList[8][1]
        
        # Draw a line between the thumb tip and index finger tip.
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # calculate the distance from thumb to finger
        length = hypot(x2 - x1, y2 - y1)

        # calculate brightness value (from 0 to 100).
        bright = np.interp(length, [15, 200], [0, 100])
        print(bright, length)
        #set the value of brightness
        sbc.set_brightness(int(bright))


    # if right hand is detected
    if right_lmList != []:
        # Get the coordinates of the thumb tip (index 4) and index finger tip (index 8) from the left hand landmarks
        x1, y1 = right_lmList[4][0], right_lmList[4][1]
        x2, y2 = right_lmList[8][0], right_lmList[8][1]

        # Draw a line between the thumb tip and index finger tip.
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # calculate the length from finger to thumb
        length = hypot(x2 - x1, y2 - y1)
        
        #calculate the value of volume
        vol = np.interp(length, [15, 200], [volMin, volMax])
        print(vol, length)

        # set the volume 
        volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
