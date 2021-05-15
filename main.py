import numpy as np
import cv2
import faceDetection as fd
import landmarkLocalization as ll

# Más bajo -< más rápido, peor detección
# Más alto -> más lento, mejor detección
# No bajar de 0 (no incluído) y no recomendable pasar de 1
DOWNSIZE_FACTOR = 0.35
videoInput = cv2.VideoCapture(0)

while True:
    ret, frame = videoInput.read()
    detected, box = fd.getBoundingBox(frame, downsizeFactor=DOWNSIZE_FACTOR)

    if detected:
        # Por si se ha hecho downsizing, se recupera el tamaño del la bb
        box = (box/DOWNSIZE_FACTOR).astype(int)
        (x,y,w,h) = box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        # Obtenemos 68 puntos de referencia de la cara detectada
        landmarksCoords = ll.getLandmarksCoordinates(frame, box)
        for coord in landmarksCoords:
            cv2.circle(frame, coord, 3, (0,0,255))

    cv2.imshow('Output', frame)

    # 27 es el código de ESC en mi MacBook Pro (puede cambiar para cada PC)
    if cv2.waitKey(1) == 27:
        break

videoInput.release()
cv2.destroyAllWindows()