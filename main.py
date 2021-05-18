import numpy as np
import cv2
import dlib
from imutils.video import FPS
import faceDetection as fd
import landmarkLocalization as ll
import poseEstimation as pe

# Más bajo -< más rápido, peor detección
# Más alto -> más lento, mejor detección
# No bajar de 0 (no incluído) y no recomendable pasar de 1
DOWNSIZE_FACTOR = 0.5
FACE_DETECTION_MODEL = "./trainedModels/haarcascade_frontalface_default.xml"
SHAPE_MODEL = "./trainedModels/shape_predictor_68_face_landmarks.dat"

faceDetector = cv2.CascadeClassifier(FACE_DETECTION_MODEL)
shapePredictor = dlib.shape_predictor(SHAPE_MODEL)
videoInput = cv2.VideoCapture(0)

ret, frame = videoInput.read()
size = frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
cameraMat = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = np.float64
                        )

# Modelo 3D estándar
modelPoints = np.array([
                            (0.0, 0.0, 0.0),             # Punta nariz
                            (0.0, -330.0, -65.0),        # Barbilla
                            (-225.0, 170.0, -135.0),     # Ojo izquierdo izquierda
                            (225.0, 170.0, -135.0),      # Ojo derecho derecha
                            (-150.0, -150.0, -125.0),    # Boca izquierda
                            (150.0, -150.0, -125.0)      # Boca derecha
                        ])

# Comenzar contador de FPS
fps = FPS().start()

while True:
    ret, frame = videoInput.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = grayFrame.shape
    height = int(height*DOWNSIZE_FACTOR)
    width = int(width*DOWNSIZE_FACTOR)
    # Reducimos (o no) el tamaño.
    resizedGrayFrame = cv2.resize(grayFrame, (int(width), int(height)))

    detected, box = fd.getBoundingBox(faceDetector, resizedGrayFrame)
    if detected:
        # Por si se ha hecho downsizing, se recupera el tamaño del la bb
        drawingBox = (box/DOWNSIZE_FACTOR).astype(np.int64)
        (x,y,w,h) = drawingBox       
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)    
        
        # Obtenemos 68 puntos de referencia de la cara detectada 
        landmarksCoords = ll.getLandmarksCoordinates(shapePredictor, resizedGrayFrame, box)
        drawingCoords = (landmarksCoords/DOWNSIZE_FACTOR).astype(np.int64)  
        for idx, coord in enumerate(drawingCoords):
            if idx in [33,8,36,45,48,54]:
                cv2.circle(frame, coord, 3, (0,255,255))
            else:
                cv2.circle(frame, coord, 1, (0,255,0))

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
        
         
    cv2.imshow('Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', frame)

    # Actualizar el contador de FPS
    fps.update()
    
    # 27 es el código de ESC en mi MacBook Pro (puede cambiar para cada PC)
    if cv2.waitKey(1) == 27:
        break

# Informe de los FPS
fps.stop()
print("[INFO] tiempo: {:.2f}".format(fps.elapsed()),"s")
print("[INFO] FPS: {:.2f}".format(fps.fps()))

videoInput.release()
cv2.destroyAllWindows()