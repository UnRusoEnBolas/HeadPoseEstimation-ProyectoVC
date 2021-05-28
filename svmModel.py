import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.load("train_features.npy")
y = np.load("./ExpressionClassification/train_labels.npy")

X = np.swapaxes(X,0,1)


clf = svm.SVC(kernel="linear")
clf.fit(X,y)


test_data = np.load("test_features.npy")

test_data = np.swapaxes(test_data,0,1)

y_pred = clf.predict(test_data)

y_test = np.load("./ExpressionClassification/test_labels.npy")


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))