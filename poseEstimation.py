import cv2
import numpy as np


inputId = 0
detThreshold = 0.55
nmsThreshold = 0.35
                     

def poseEstimation(frame, face, drawingBox, landmarks):
    
    # x = startX
    # y = startY
    # x+w = endX
    # y+h = endY
    

    #Preparamos los puntos claves de los landmarks
    image_points = np.array([
                (landmarks[33,0]*widthX, landmarks[33,1]*heightY), # Nariz
                (landmarks[8,0]*widthX, landmarks[8,1]*heightY), # Barbilla
                (landmarks[36,0]*widthX, landmarks[36,1]*heightY), # Extremo ojo izquierdo
                (landmarks[45,0]*widthX, landmarks[45,1]*heightY), # Extremo ojo derecho
                (landmarks[48,0]*widthX, landmarks[48,1]*heightY), # Extremo izquierdo boca
                (landmarks[54,0]*widthX, landmarks[54,1]*heightY)  # Extremo derecho boca
    ], dtype="double")
    

