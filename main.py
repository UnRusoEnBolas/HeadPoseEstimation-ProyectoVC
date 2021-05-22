import numpy as np
import cv2
import sys
import mediapipe as mp 
from imutils.video import FPS
import poseEstimation as pe
from processFile import get3DFaceModelAsArray

ENABLE_POSE = False
AXIS_SCALE = 50
AXIS_POSITION_OFFSET = (50,70)

for args in sys.argv:
    if args == "--pose":
        ENABLE_POSE = True

mediaPipeFaceMesh = mp.solutions.face_mesh
img = cv2.imread("./imgTest.png")

size = img.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
cameraMat = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype=np.float64
)

modelPoints = get3DFaceModelAsArray('canonicalDaceModel_Coordinates.txt')

with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
    img.flags.writeable = False
    res = faceMesh.process(img)
    img.flags.writeable = True

    landmarksCoords = np.empty((468, 2), dtype=np.float64)
    if res.multi_face_landmarks:
        while True:
                for detectedLandmarks in res.multi_face_landmarks:
                    for i in range(0, 468):
                        landmarksCoords[i] = [
                            detectedLandmarks.landmark[i].x,
                            detectedLandmarks.landmark[i].y
                        ]
                landmarksCoords = landmarksCoords * np.array([[size[1], size[0]]])
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

                    


                cv2.imshow(
                'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', img)

                if cv2.waitKey(1) == 27:
                    break
cv2.destroyAllWindows()
