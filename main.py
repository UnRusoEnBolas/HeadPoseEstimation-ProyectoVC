import numpy as np
import cv2
import dlib
import faceDetection as fd
import landmarkLocalization as ll

# Más bajo -< más rápido, peor detección
# Más alto -> más lento, mejor detección
# No bajar de 0 (no incluído) y no recomendable pasar de 1
DOWNSIZE_FACTOR = 0.75
FACE_DETECTION_MODEL = "./trainedModels/haarcascade_frontalface_default.xml"
SHAPE_MODEL = "./trainedModels/shape_predictor_68_face_landmarks.dat"

faceDetector = cv2.CascadeClassifier(FACE_DETECTION_MODEL)
shapePredictor = dlib.shape_predictor(SHAPE_MODEL)
videoInput = cv2.VideoCapture(0)

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
         
    cv2.imshow('Head pose estimation by Juan Carlos Soriano and Jorge Gimenez', frame)

    # 27 es el código de ESC en mi MacBook Pro (puede cambiar para cada PC)
    if cv2.waitKey(1) == 27:
        break

videoInput.release()
cv2.destroyAllWindows()