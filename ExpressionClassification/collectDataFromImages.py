import numpy as np
import cv2
import mediapipe as mp
import os

IMAGES_PATH = "./dataset"

nImages = len(os.listdir(IMAGES_PATH))
landmarksCoords = np.empty((468, 2, nImages), dtype=np.float64)

mediaPipeFaceMesh = mp.solutions.face_mesh

for idx, imageFilename in enumerate(os.listdir(IMAGES_PATH)):
    print(IMAGES_PATH + '/' + imageFilename)
    img = cv2.imread(IMAGES_PATH + '/' + imageFilename)
    cv2.imshow('mage', img)
    cv2.waitKey(0)

    size = img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
        img.flags.writeable = False
        res = faceMesh.process(img)
        img.flags.writeable = True

        if res.multi_face_landmarks:
            for detectedLandmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    landmarksCoords[:,:,idx] = [
                        detectedLandmarks.landmark[i].x,
                        detectedLandmarks.landmark[i].y
                    ]
            landmarksCoords[:,:,idx] = landmarksCoords[:,:,idx] * np.array([[size[1], size[0]]])
print(landmarksCoords)