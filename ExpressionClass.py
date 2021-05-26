from typing import NoReturn
import numpy as np
import cv2
import sys
import os.path
import mediapipe as mp
import poseEstimation as pe 
from processFile import get3DFaceModelAsArray
import random

from PIL import Image
import PIL

ENABLE_POSE = False
ENABLE_VIEW = False
ENABLE_TRAIN = False
ENABLE_TEST = False

AXIS_SCALE = 50
AXIS_POSITION_OFFSET = (50,70)

for args in sys.argv:
    if args == "--pose":
        ENABLE_POSE = True

mediaPipeFaceMesh = mp.solutions.face_mesh

size = (48,48)
focal_length = size[1]
center = (size[1]/2, size[0]/2)
cameraMat = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype=np.float64
)

modelPoints = get3DFaceModelAsArray('canonicalDaceModel_Coordinates.txt')

train_files = np.load("./ExpressionClassification/test_images.npy")
test_files = np.load("./ExpressionClassification/test_images.npy")
train_landmarks = np.empty((468,2,train_files.shape[2]))
test_landmarks = np.empty((468,2,test_files.shape[2]))

if os.path.isfile("train_landmarks.npy"):
    print("Train trobat!")
else:
    print("Train NO trobat!")
    ENABLE_TRAIN = True



if ENABLE_TRAIN:
    for idx in range(train_files.shape[2]):
        img = train_files[:,:,idx]
        newDims = img.shape[0]*5
        img = cv2.resize(img, dsize=(newDims, newDims), interpolation=cv2.INTER_CUBIC)
        img = np.repeat(img[:,:,np.newaxis],3,axis=2)  
        size = img.shape

        with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
            img.flags.writeable = False
            res = faceMesh.process(img)
            img.flags.writeable = True

            landmarksCoords = np.empty((468, 2), dtype=np.float64)
            if res.multi_face_landmarks:
                    for detectedLandmarks in res.multi_face_landmarks:
                        for i in range(0, 468):
                            landmarksCoords[i] = [
                                detectedLandmarks.landmark[i].x,
                                detectedLandmarks.landmark[i].y
                            ]
                    landmarksCoords = landmarksCoords * np.array([[size[1], size[0]]])

                    train_landmarks[:,:,idx] = landmarksCoords
                    
                    if ENABLE_VIEW:
                        for landmarkCoord in landmarksCoords.astype(np.int64):
                                cv2.circle(img, landmarkCoord, 1, (0,255,0))

                    if ENABLE_POSE:
                        success, rotVector, traVect = pe.poseEstimation(modelPoints, cameraMat, landmarksCoords)

                        rotMat, _ = cv2.Rodrigues(rotVector)
                        axis = (np.float64([[1,0,0], [0,1,0], [0,0,1]])*AXIS_SCALE)
                        axis = (rotMat @ axis.T).T + [AXIS_POSITION_OFFSET[0],AXIS_POSITION_OFFSET[1],0]
                        for i in range(3):
                            p1 = AXIS_POSITION_OFFSET
                            p2 = (axis[i][0].astype(np.int64), axis[i][1].astype(np.int64))
                            color = (0,0,255) if i == 0 else (0,255,0) if i == 1 else (255,0,0)
                            cv2.line(img, p1, p2, color, 2)
                        
                        blk = np.zeros(img.shape, np.uint8)
                        cv2.rectangle(blk, (5, 5), (120, 120), (255,255,255), cv2.FILLED)
                        comb = cv2.addWeighted(img, 1.0, blk, 0.35, 1)

                        
                        (noseTip, _) = cv2.projectPoints(np.array([(0.0, 0.0, 20.0)]), rotVector, traVect, cameraMat, np.zeros((4,1)))
                        p1 = (int(landmarksCoords[1][0]),int(landmarksCoords[1][1]))
                        p2 = (int(noseTip[0][0][0]), int(noseTip[0][0][1]))
                        cv2.line(comb, p1, p2, (255,0,0), 2)

                        cv2.imshow(
                    'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', comb)

                        

                    if ENABLE_VIEW:
                        cv2.imshow(
                        'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', img)

                        if cv2.waitKey(1) == 27:
                            break
        cv2.destroyAllWindows()

    np.save('train_landmarks.npy', train_landmarks)

else:
    train_landmarks = np.load("./train_landmarks.npy")



if ENABLE_TEST:
    for idx in range(test_files.shape[2]):
        img = test_files[:,:,idx]
        newDims = img.shape[0]*5
        img = cv2.resize(img, dsize=(newDims, newDims), interpolation=cv2.INTER_CUBIC)
        img = np.repeat(img[:,:,np.newaxis],3,axis=2)  
        size = img.shape

        with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
            img.flags.writeable = False
            res = faceMesh.process(img)
            img.flags.writeable = True

            landmarksCoords = np.empty((468, 2), dtype=np.float64)
            if res.multi_face_landmarks:
                    for detectedLandmarks in res.multi_face_landmarks:
                        for i in range(0, 468):
                            landmarksCoords[i] = [
                                detectedLandmarks.landmark[i].x,
                                detectedLandmarks.landmark[i].y
                            ]
                    landmarksCoords = landmarksCoords * np.array([[size[1], size[0]]])

                    test_landmarks[:,:,idx] = landmarksCoords
                    
                    if ENABLE_VIEW:
                        for landmarkCoord in landmarksCoords.astype(np.int64):
                                cv2.circle(img, landmarkCoord, 1, (0,255,0))

                    if ENABLE_POSE:
                        success, rotVector, traVect = pe.poseEstimation(modelPoints, cameraMat, landmarksCoords)

                        rotMat, _ = cv2.Rodrigues(rotVector)
                        axis = (np.float64([[1,0,0], [0,1,0], [0,0,1]])*AXIS_SCALE)
                        axis = (rotMat @ axis.T).T + [AXIS_POSITION_OFFSET[0],AXIS_POSITION_OFFSET[1],0]
                        for i in range(3):
                            p1 = AXIS_POSITION_OFFSET
                            p2 = (axis[i][0].astype(np.int64), axis[i][1].astype(np.int64))
                            color = (0,0,255) if i == 0 else (0,255,0) if i == 1 else (255,0,0)
                            cv2.line(img, p1, p2, color, 2)
                        
                        blk = np.zeros(img.shape, np.uint8)
                        cv2.rectangle(blk, (5, 5), (120, 120), (255,255,255), cv2.FILLED)
                        comb = cv2.addWeighted(img, 1.0, blk, 0.35, 1)

                        
                        (noseTip, _) = cv2.projectPoints(np.array([(0.0, 0.0, 20.0)]), rotVector, traVect, cameraMat, np.zeros((4,1)))
                        p1 = (int(landmarksCoords[1][0]),int(landmarksCoords[1][1]))
                        p2 = (int(noseTip[0][0][0]), int(noseTip[0][0][1]))
                        cv2.line(comb, p1, p2, (255,0,0), 2)

                        cv2.imshow(
                    'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', comb)

                        

                    if ENABLE_VIEW:
                        cv2.imshow(
                        'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', img)

                        if cv2.waitKey(1) == 27:
                            break
        cv2.destroyAllWindows()

    np.save('test_landmarks.npy', test_landmarks)

else:
    test_landmarks = np.load("./test_landmarks.npy")

