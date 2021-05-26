import numpy as np
import cv2
import mediapipe as mp
from imutils.video import FPS
import poseEstimation as pe
from processFile import get3DFaceModelAsArray

AXIS_SCALE = 50
AXIS_POSITION_OFFSET = (50,70)

mediaPipeFaceMesh = mp.solutions.face_mesh
videoInput = cv2.VideoCapture(0)

ret, frame = videoInput.read()
size = frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
cameraMat = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype=np.float64
)

modelPoints = get3DFaceModelAsArray('canonicalDaceModel_Coordinates.txt')

# Comenzar contador de FPS
fps = FPS().start()

with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
    while True:
        ret, frame = videoInput.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgbFrame.flags.writeable = False
        res = faceMesh.process(rgbFrame)
        rgbFrame.flags.writeable = True

        landmarksCoords = np.empty((468, 2), dtype=np.float64)
        if res.multi_face_landmarks:
            for detectedLandmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    landmarksCoords[i] = [
                        detectedLandmarks.landmark[i].x,
                        detectedLandmarks.landmark[i].y
                    ]
            landmarksCoords = landmarksCoords * np.array([[size[1], size[0]]])
            counter = 0
            for landmarkCoord in landmarksCoords.astype(np.int64):
                if counter == 1:
                    cv2.circle(frame, landmarkCoord, 3, (0,0,255))
                else:
                    cv2.circle(frame, landmarkCoord, 1, (0,255,0))
                counter = counter + 1

            success, rotVector, traVect = pe.poseEstimation(modelPoints, cameraMat, landmarksCoords)

            rotMat, _ = cv2.Rodrigues(rotVector)
            axis = (np.float64([[1,0,0], [0,1,0], [0,0,1]])*AXIS_SCALE)
            axis = (rotMat @ axis.T).T + [AXIS_POSITION_OFFSET[0],AXIS_POSITION_OFFSET[1],0]
            for i in range(3):
                p1 = AXIS_POSITION_OFFSET
                p2 = (axis[i][0].astype(np.int64), axis[i][1].astype(np.int64))
                color = (0,0,255) if i == 0 else (0,255,0) if i == 1 else (255,0,0)
                cv2.line(frame, p1, p2, color, 2)
            
            (noseTip, _) = cv2.projectPoints(np.array([(0.0, 0.0, 20.0)]), rotVector, traVect, cameraMat, np.zeros((4,1)))
            p1 = (int(landmarksCoords[1][0]),int(landmarksCoords[1][1]))
            p2 = (int(noseTip[0][0][0]), int(noseTip[0][0][1]))
            cv2.line(frame, p1, p2, (255,0,0), 2)

        cv2.imshow(
            'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', frame)
        fps.update()

        # 27 es el código de ESC en mi MacBook Pro (puede cambiar para cada PC)
        if cv2.waitKey(1) == 27:
            break

# Informe de los FPS
fps.stop()
print("[INFO] tiempo: {:.2f}".format(fps.elapsed()), "s")
print("[INFO] FPS: {:.2f}".format(fps.fps()))

videoInput.release()
cv2.destroyAllWindows()
