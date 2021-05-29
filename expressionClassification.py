import numpy as np
import math
from sklearn import preprocessing

def getExpression(classificationModel, landmarks, expressionList):
    featuresArr = np.empty((12), dtype=np.float64)

    featuresArr[0] = round(math.dist(landmarks[145,:],landmarks[159,:]),2)
    featuresArr[1] = round(math.dist(landmarks[133,:],landmarks[33,:]),2)
    featuresArr[2] = round(math.dist(landmarks[374,:],landmarks[386,:]),2)
    featuresArr[3] = round(math.dist(landmarks[263,:],landmarks[362,:]),2)
    featuresArr[4] = round(math.dist(landmarks[70,:],landmarks[55,:]),2)
    featuresArr[5] = round(math.dist(landmarks[300,:],landmarks[285,:]),2)
    featuresArr[6] = round(math.dist(landmarks[291,:],landmarks[61,:]),2)
    featuresArr[7] = round(math.dist(landmarks[159,:],landmarks[52,:]),2)
    featuresArr[8] = round(math.dist(landmarks[386,:],landmarks[282,:]),2)
    featuresArr[9] = round(math.dist(landmarks[13,:],landmarks[1,:]),2)
    featuresArr[10] = round(math.dist(landmarks[61,:],landmarks[145,:]),2)
    featuresArr[11] = round(math.dist(landmarks[291,:],landmarks[374,:]),2)

    featuresArr = featuresArr.reshape(1,12)
    # Se tendrían que normalizar
    
    res = classificationModel(featuresArr)
    return expressionList[np.argmax(res)]