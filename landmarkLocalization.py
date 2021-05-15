import numpy as np
import cv2
import dlib

def getLandmarksCoordinates(frame, box, trainedModelPath="./trainedModels/shape_predictor_68_face_landmarks.dat"):
    # Pasamos a escala de grises 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    shapePredictor = dlib.shape_predictor(trainedModelPath)
    rectangle = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
    # Conseguimos las coordenadas de los distintos landmarks
    shape = shapePredictor(frame, rectangle)

    coordinates = np.empty((shape.num_parts,2), dtype=np.int64)
    for i in range(shape.num_parts):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates