import numpy as np
import cv2
import mediapipe as mp
from imutils.video import FPS

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

# Modelo 3D estándar
modelPoints = np.array([
    (0.0, 0.0, 0.0),             # Punta nariz
    (0.0, -330.0, -65.0),        # Barbilla
    # Ojo izquierdo izquierda
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),      # Ojo derecho derecha
    (-150.0, -150.0, -125.0),    # Boca izquierda
    (150.0, -150.0, -125.0)      # Boca derecha
])

# Comenzar contador de FPS
fps = FPS().start()

with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
    while True:
        ret, frame = videoInput.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ------------------------------TESTING----------------------------------
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
            for landmarkCoord in landmarksCoords.astype(np.int64):
                cv2.circle(frame, landmarkCoord, 1, (0,255,0))
        # -----------------------------------------------------------------------

        '''
        # Obtenemos la estimation pose

        # Preparamos los puntos claves de los landmarks
        poseEstimationLandmarks = np.array([
                    (landmarksCoords[33,0], landmarksCoords[33,1]), # Nariz
                    (landmarksCoords[8,0], landmarksCoords[8,1]), # Barbilla
                    (landmarksCoords[36,0], landmarksCoords[36,1]), # Extremo ojo izquierdo
                    (landmarksCoords[45,0], landmarksCoords[45,1]), # Extremo ojo derecho
                    (landmarksCoords[48,0], landmarksCoords[48,1]), # Extremo izquierdo boca
                    (landmarksCoords[54,0], landmarksCoords[54,1])  # Extremo derecho boca
        ], dtype=np.float64)

        success, rotVector, traVect, _ = pe.poseEstimation(modelPoints, cameraMat, poseEstimationLandmarks)

        
        poseEstimationLandmarks = poseEstimationLandmarks/DOWNSIZE_FACTOR
        (noseTip, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotVector, traVect, cameraMat, np.zeros((4,1)))
        # Dibujamos la linea de la pose
        p1 = (int(poseEstimationLandmarks[0][0]),int(poseEstimationLandmarks[0][1]))
        p2 = (int(noseTip[0][0][0]/DOWNSIZE_FACTOR), int(noseTip[0][0][1]/DOWNSIZE_FACTOR))
        cv2.line(frame, p1, p2, (0,0,255), 2)
        '''

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
