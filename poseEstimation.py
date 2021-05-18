import cv2
import numpy as np


inputId = 0
detThreshold = 0.55
nmsThreshold = 0.35
                     

def poseEstimation(model_points, camera_matrix, image_points):

    # return cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    return cv2.solvePnPRansac(model_points, image_points, camera_matrix, np.zeros((4,1)))
    """return (success, rotation_vector, translation_vector)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Dibujamos la linea de la pose
    p1 = ( widthX+int(image_points[0][0]), heightY+int(image_points[0][1]))
    p2 = ( widthX+int(nose_end_point2D[0][0][0]), heightY+int(nose_end_point2D[0][0][1]))"""
    



    

