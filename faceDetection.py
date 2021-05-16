import cv2

def getBoundingBox(faceDetector, frame):
    # Obtenemos los rectángulos de las caras detectadas.
    faces = faceDetector.detectMultiScale(frame, minNeighbors=6 ,minSize=(30,30))
    
    if len(faces) > 0:
        return True, faces
    else:
        return False, None
