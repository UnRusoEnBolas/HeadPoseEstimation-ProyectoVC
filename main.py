import numpy as np
import cv2
import mediapipe as mp
from imutils.video import FPS
import poseEstimation as pe
from processFile import get3DFaceModelAsArray

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
            for idx, landmarkCoord in enumerate(landmarksCoords.astype(np.int64)):
                if idx in [94, 152, 33, 263, 61, 291]:
                    cv2.circle(frame, landmarkCoord, 3, (0,255,255))
                else:
                    cv2.circle(frame, landmarkCoord, 1, (0,255,0))

            success, rotVector, traVect = pe.poseEstimation(modelPoints, cameraMat, landmarksCoords)

            (noseTip, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotVector, traVect, cameraMat, np.zeros((4,1)))
            # Dibujamos la linea de la pose
            p1 = (int(landmarksCoords[94][0]),int(landmarksCoords[94][1]))
            p2 = (int(noseTip[0][0][0]), int(noseTip[0][0][1]))
            cv2.line(frame, p1, p2, (0,0,255), 2)

        cv2.imshow(
            'Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', frame)

        # Actualizar el contador de FPS
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
