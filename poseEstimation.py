import cv2
import numpy as np


inputId = 0
detThreshold = 0.55
nmsThreshold = 0.35
                     

def poseEstimation(model_points, camera_matrix, image_points):

    return cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    # return cv2.solvePnPRansac(model_points, image_points, camera_matrix, np.zeros((4,1)))
    



    

