from sklearn.model_selection import train_test_split
import numpy as np

X = np.load("./ExpressionClassification/landmarks.npy")
y = np.load("./ExpressionClassification/labels.npy")

X = np.swapaxes(X,0,1)
X = np.swapaxes(X,0,2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

X_train = np.swapaxes(X_train,0,2)
X_train = np.swapaxes(X_train,0,1)
X_test = np.swapaxes(X_test,0,2)
X_test = np.swapaxes(X_test,0,1)

np.save("./ExpressionClassification/train_landmarks.npy", X_train)
np.save("./ExpressionClassification/train_labels.npy", y_train)
np.save("./ExpressionClassification/test_landmarks.npy",X_test)
np.save("./ExpressionClassification/test_labels.npy",y_test)

