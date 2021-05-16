import numpy as np
import cv2
import dlib

def getLandmarksCoordinates(frame, box, shapePredictor, downsizeFactor=1):#trainedModelPath="./trainedModels/shape_predictor_68_face_landmarks.dat"):
    # Pasamos a escala de grises 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = frame.shape
    height = int(height*downsizeFactor)
    width = int(width*downsizeFactor)
    # Reducimos (o no) el tamaño.
    frame = cv2.resize(frame, (int(width), int(height)))
    box = (box*downsizeFactor).astype(int)

    #shapePredictor = dlib.shape_predictor(trainedModelPath)
    rectangle = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
    # Conseguimos las coordenadas de los distintos landmarks
    shape = shapePredictor(frame, rectangle)

    coordinates = np.empty((shape.num_parts,2), dtype=np.int64)
    for i in range(shape.num_parts):
        coordinates[i] = (shape.part(i).x/downsizeFactor, shape.part(i).y/downsizeFactor)

    return coordinates