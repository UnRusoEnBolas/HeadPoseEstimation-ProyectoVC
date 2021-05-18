import cv2
import numpy as np


inputId = 0
detThreshold = 0.55
nmsThreshold = 0.35
                     

def poseEstimation(frame, face, drawingBox, landmarks):

    (x,y,w,h) = drawingBox
    widthX = x+w
    heightY = y+h
    

    # Preparamos los puntos claves de los landmarks
    image_points = np.array([
                (landmarks[33,0]*widthX, landmarks[33,1]*heightY), # Nariz
                (landmarks[8,0]*widthX, landmarks[8,1]*heightY), # Barbilla
                (landmarks[36,0]*widthX, landmarks[36,1]*heightY), # Extremo ojo izquierdo
                (landmarks[45,0]*widthX, landmarks[45,1]*heightY), # Extremo ojo derecho
                (landmarks[48,0]*widthX, landmarks[48,1]*heightY), # Extremo izquierdo boca
                (landmarks[54,0]*widthX, landmarks[54,1]*heightY)  # Extremo derecho boca
    ], dtype="double")

    # Estimamos posiciones de las puntos característicos
    eye_center_x = (image_points[2][0]+image_points[3][0])/2
    eye_center_y = (image_points[2][1]+image_points[3][1])/2
    nose_offset_x = (image_points[0][0] - eye_center_x)
    nose_offset_y = (image_points[0][1] - eye_center_y)
    mouth_center_x = (image_points[4][0] + image_points[5][0])/2
    mouth_center_y = (image_points[4][1] + image_points[5][1])/2

    

