import numpy as np
import cv2

videoInput = cv2.VideoCapture(0)

while True:
    ret, frame = videoInput.read()
    cv2.imshow('VideoInput', frame)
    if cv2.waitKey(1) == 27: # 27 es el código de ESC en mi MacBook
        break

videoInput.release()
cv2.destroyAllWindows()