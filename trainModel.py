import numpy as np

train_x = np.load("train_features.npy")
train_y = np.load("ExpressionClassification/train_labels.npy")

test_x = np.load("test_features.npy")
test_y = np.load("ExpressionClassification/test_labels.npy")

print(train_x.shape)
print(test_x.shape)