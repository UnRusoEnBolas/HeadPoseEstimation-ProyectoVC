import numpy as np
import cv2
import mediapipe as mp

mediaPipeFaceMesh = mp.solutions.face_mesh
videoInput = cv2.VideoCapture(0)
_, frame = videoInput.read()

HEIGHT, WIDTH, _ = frame.shape
EMOTIONS = ["NEUTRAL", "HAPPY", "ANGRY", "SAD", "SURPRISED"]
EMOTION = 0

with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:
    while True:
        ret, frame = videoInput.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgbFrame.flags.writeable = False
        res = faceMesh.process(rgbFrame)
        rgbFrame.flags.writeable = True

        landmarksCoords = np.empty((468, 2), dtype=np.float64)
        if res.multi_face_landmarks:
            for detectedLandmarks in res.multi_face_landmarks:
                for i in range(0, 468):
                    landmarksCoords[i] = [
                        detectedLandmarks.landmark[i].x,
                        detectedLandmarks.landmark[i].y
                    ]
            landmarksCoords = landmarksCoords * np.array([[WIDTH, HEIGHT]])
            for landmarkCoord in landmarksCoords.astype(np.int64):
                cv2.circle(frame, landmarkCoord, 1, (0,255,0))
        
        cv2.putText(frame, f"LABEL: {EMOTIONS[EMOTION]}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.imshow('Data collection program', frame)

        if cv2.waitKey(1) == 27:
            break
        
        if cv2.waitKey(50) == 32:
            # Guardar landmarks y label
            print("Spacebar")
        
        if cv2.waitKey(50) == 101:
            print("e")
            EMOTION += 1
            if EMOTION >= len(EMOTIONS):
                EMOTION = 0

videoInput.release()
cv2.destroyAllWindows()
