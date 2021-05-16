import numpy as np
import cv2
import dlib

def getLandmarksCoordinates(shapePredictor, frame, box):
    rectangle = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
    # Conseguimos las coordenadas de los distintos landmarks
    shape = shapePredictor(frame, rectangle)

    coordinates = np.empty((shape.num_parts,2), dtype=np.int64)
    for i in range(shape.num_parts):
        # coordinates[i] = (shape.part(i).x/downsizeFactor, shape.part(i).y/downsizeFactor)
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates