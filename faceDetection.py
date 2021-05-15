import cv2

def getBoundingBox(frame, downsizeFactor=1, trainedModelPath="./trainedModels/haarcascade_frontalface_default.xml"):
    # Pasamos a escala de grises.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = frame.shape
    height = int(height*downsizeFactor)
    width = int(width*downsizeFactor)
    # Reducimos (o no) el tamaño.
    frame = cv2.resize(frame, (int(width), int(height)))

    faceCascadeDetector = cv2.CascadeClassifier(trainedModelPath)
    # Obtenemos los rectángulos de las caras detectadas.
    faces = faceCascadeDetector.detectMultiScale(frame)
    
    if len(faces) > 0:
        return True, faces[0]
    else:
        return False, None
