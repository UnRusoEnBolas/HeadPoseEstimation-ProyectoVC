import cv2

def getBoundingBox(faceDetector, frame):
    # Obtenemos los rectángulos de las caras detectadas.
    faces = faceDetector.detectMultiScale(frame)
    
    if len(faces) > 0:
        return True, faces[0]
    else:
        return False, None
