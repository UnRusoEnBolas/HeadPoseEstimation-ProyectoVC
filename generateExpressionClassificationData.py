from typing import NoReturn
import numpy as np
import cv2
import sys
import os.path
import mediapipe as mp
import poseEstimation as pe 
from processFile import get3DFaceModelAsArray
import math

from PIL import Image
import PIL

ENABLE_POSE = False
ENABLE_VIEW = False
ENABLE_TRAIN = False
ENABLE_TEST = False
ENABLE_FEATURES = True

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

train_files = np.load("./ExpressionClassification/train_images.npy")
test_files = np.load("./ExpressionClassification/test_images.npy")
train_landmarks = np.empty((468,2,train_files.shape[2]))
test_landmarks = np.empty((468,2,test_files.shape[2]))

if os.path.isfile("./ExpressionClassification/train_landmarks.npy"):
    print("Train trobat!")
else:
    print("Train NO trobat!")
    ENABLE_TRAIN = True

if os.path.isfile("./ExpressionClassification/test_landmarks.npy"):
    print("Test trobat!")
else:
    print("Test NO trobat!")
    ENABLE_TEST = True

if os.path.isfile("./ExpressionClassification/train_features.npy"):
    print("Features trobat!")
else:
    print("Features NO trobat!")
    ENABLE_FEATURES = True



if ENABLE_TRAIN:
    for idx in range(train_files.shape[2]):
        img = train_files[:,:,idx]
        #newDims = img.shape[0]*5
        #img = cv2.resize(img, dsize=(newDims, newDims), interpolation=cv2.INTER_CUBIC)
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

    np.save('./ExpressionClassification/train_landmarks.npy', train_landmarks)

else:
    train_landmarks = np.load("./ExpressionClassification/train_landmarks.npy")



if ENABLE_TEST:
    for idx in range(test_files.shape[2]):
        img = test_files[:,:,idx]
        #newDims = img.shape[0]*5
        #img = cv2.resize(img, dsize=(newDims, newDims), interpolation=cv2.INTER_CUBIC)
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

    np.save('./ExpressionClassification/test_landmarks.npy', test_landmarks)

else:
    test_landmarks = np.load("./ExpressionClassification/test_landmarks.npy")


if ENABLE_FEATURES:
    #landmarks features:
        #F1 Altura ojo izquierdo (145-159)
        #F2 Anchura ojo izquierdo (133-33)
        #F3 Altura ojo derecho (374-386)
        #F4 Anchura ojo derecho (263-362)
        #F5 Anchura ceja izquierda (70-55)
        #F6 Anchura ceja derecha (300-285)
        #F7 Anchura Labios (291-61)
        #F8 Distancia entre la parte superior del centro del ojo izquierdo
         #con el centro de la ceja izquierda (159-52)
        #F9 Distancia entre la parte superior del centro del ojo derecho
         #con el centro de la ceja derecha (386-282)
        #F10 Distancia entre el centro de la nariz y el centro de los labios (13-1)
        #F11 Distancia entre la parte inferior del centro del ojo izquierdo 
         #con el extremo izquierdo de los labios (61-145)
        #F12 Distancia entre la parte inferior del centro del ojo derecho
         #con el extremo derecho de los labios (291-374)
    
    array_features_train = np.empty((12,train_landmarks.shape[2]))
    array_features_test = np.empty((12,test_landmarks.shape[2]))

    
    
    #Obtenemos las features de los datos de train y de test (468,2,7178)
    #SUPONEMOS QUE TRAIN Y TEST TIENEN EL MISMO NUMERO DE DATOS

    for idx in range(train_landmarks.shape[2]):
        array_features_train[0,idx] = round(math.dist(train_landmarks[145,:,idx],train_landmarks[159,:,idx]),2)
        array_features_train[1,idx] = round(math.dist(train_landmarks[133,:,idx],train_landmarks[33,:,idx]),2)
        array_features_train[2,idx] = round(math.dist(train_landmarks[374,:,idx],train_landmarks[386,:,idx]),2)
        array_features_train[3,idx] = round(math.dist(train_landmarks[263,:,idx],train_landmarks[362,:,idx]),2)
        array_features_train[4,idx] = round(math.dist(train_landmarks[70,:,idx],train_landmarks[55,:,idx]),2)
        array_features_train[5,idx] = round(math.dist(train_landmarks[300,:,idx],train_landmarks[285,:,idx]),2)
        array_features_train[6,idx] = round(math.dist(train_landmarks[291,:,idx],train_landmarks[61,:,idx]),2)
        array_features_train[7,idx] = round(math.dist(train_landmarks[159,:,idx],train_landmarks[52,:,idx]),2)
        array_features_train[8,idx] = round(math.dist(train_landmarks[386,:,idx],train_landmarks[282,:,idx]),2)
        array_features_train[9,idx] = round(math.dist(train_landmarks[13,:,idx],train_landmarks[1,:,idx]),2)
        array_features_train[10,idx] = round(math.dist(train_landmarks[61,:,idx],train_landmarks[145,:,idx]),2)
        array_features_train[11,idx] = round(math.dist(train_landmarks[291,:,idx],train_landmarks[374,:,idx]),2)

    for idx in range(test_landmarks.shape[2]):
        array_features_test[0,idx] = round(math.dist(test_landmarks[145,:,idx],test_landmarks[159,:,idx]),2)
        array_features_test[1,idx] = round(math.dist(test_landmarks[133,:,idx],test_landmarks[33,:,idx]),2)
        array_features_test[2,idx] = round(math.dist(test_landmarks[374,:,idx],test_landmarks[386,:,idx]),2)
        array_features_test[3,idx] = round(math.dist(test_landmarks[263,:,idx],test_landmarks[362,:,idx]),2)
        array_features_test[4,idx] = round(math.dist(test_landmarks[70,:,idx],test_landmarks[55,:,idx]),2)
        array_features_test[5,idx] = round(math.dist(test_landmarks[300,:,idx],test_landmarks[285,:,idx]),2)
        array_features_test[6,idx] = round(math.dist(test_landmarks[291,:,idx],test_landmarks[61,:,idx]),2)
        array_features_test[7,idx] = round(math.dist(test_landmarks[159,:,idx],test_landmarks[52,:,idx]),2)
        array_features_test[8,idx] = round(math.dist(test_landmarks[386,:,idx],test_landmarks[282,:,idx]),2)
        array_features_test[9,idx] = round(math.dist(test_landmarks[13,:,idx],test_landmarks[1,:,idx]),2)
        array_features_test[10,idx] = round(math.dist(test_landmarks[61,:,idx],test_landmarks[145,:,idx]),2)
        array_features_test[11,idx] = round(math.dist(test_landmarks[291,:,idx],test_landmarks[374,:,idx]),2)

    np.save('./ExpressionClassification/train_features.npy', array_features_train)
    np.save('./ExpressionClassification/test_features.npy', array_features_test)
else:
    array_train_features = np.load("./ExpressionClassification/train_features.npy")
    array_test_features = np.load("./ExpressionClassification/test_features.npy")



        

