import numpy as np
import cv2
import dlib
import pathlib
import math
import faceDetection as fd
import landmarkLocalization as ll
import poseEstimation as pe

from imutils.video import FPS
from utils import get_child_subgraph_dpu

# Más bajo -< más rápido, peor detección
# Más alto -> más lento, mejor detección
# No bajar de 0 (no incluído) y no recomendable pasar de 1
DOWNSIZE_FACTOR = 1
FACE_DETECTION_MODEL = "./trainedModels/haarcascade_frontalface_default.xml"
SHAPE_MODEL = "./trainedModels/shape_predictor_68_face_landmarks.dat"

detThreshold = 0.55
nmsThreshold = 0.35

faceDetector = cv2.CascadeClassifier(FACE_DETECTION_MODEL)
shapePredictor = dlib.shape_predictor(SHAPE_MODEL)
videoInput = cv2.VideoCapture(0)

# Guardamos los valores de la cámara

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

ret, frame = cam.read()
size = frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )

# Modelo 3D estándar
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
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

    detected, boxes = fd.getBoundingBox(faceDetector, resizedGrayFrame)
    if detected:
        for box in boxes:
            # Por si se ha hecho downsizing, se recupera el tamaño del la bb
            drawingBox = (box/DOWNSIZE_FACTOR).astype(np.int64)
            (x,y,w,h) = drawingBox       
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)    
            
            # Obtenemos 68 puntos de referencia de la cara detectada 
            landmarksCoords = ll.getLandmarksCoordinates(shapePredictor, resizedGrayFrame, box)
            drawingCoords = (landmarksCoords/DOWNSIZE_FACTOR).astype(np.int64)  
            for coord in drawingCoords:
                cv2.circle(frame, coord, 1, (0,255,0))

            # Obtenemos la estimation pose

            # Preparamos los puntos claves de los landmarks
            image_points = np.array([
                        (landmarksCoords[33,0], landmarksCoords[33,1]), # Nariz
                        (landmarksCoords[8,0], landmarksCoords[8,1]), # Barbilla
                        (landmarksCoords[36,0], landmarksCoords[36,1]), # Extremo ojo izquierdo
                        (landmarksCoords[45,0], landmarksCoords[45,1]), # Extremo ojo derecho
                        (landmarksCoords[48,0], landmarksCoords[48,1]), # Extremo izquierdo boca
                        (landmarksCoords[54,0], landmarksCoords[54,1])  # Extremo derecho boca
            ], dtype="float")

            success, rotation_vector, translation_vector = pe.poseEstimation(model_points, camera_matrix, image_points)
            image_points = image_points / DOWNSIZE_FACTOR
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, np.zeros((4,1)))
            print(nose_end_point2D)
             # Dibujamos la linea de la pose
            p1 = (int(image_points[0][0]),int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]/DOWNSIZE_FACTOR), int(nose_end_point2D[0][0][1]/DOWNSIZE_FACTOR))
            cv2.line(frame, p1, p2, (0,0,255), 2)
            #cv2.line(frame, (int(image_points[0][0]),int(image_points[0][1])), (int(image_points[1][0]),int(image_points[1][1])), (255,0,0), 2)
         
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