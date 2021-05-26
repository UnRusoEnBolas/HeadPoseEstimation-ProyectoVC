import numpy as np
import cv2


open_files = np.load("./ExpressionClassification/test_images.npy")
open_files.astype("uint8")
print(open_files[:,:,0].shape)
cv2.imshow("...",open_files[:,:,1])

cv2.waitKey(0)
cv2.destroyAllWindows()

