import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

train_x = np.load("./ExpressionClassification/train_features.npy")
train_y = np.load("./ExpressionClassification/train_labels.npy")

test_x = np.load("./ExpressionClassification/test_features.npy")
test_y = np.load("./ExpressionClassification/test_labels.npy")

########################## Data preparation ####################################
aux = np.zeros((train_y.shape[0], int(np.max(train_y)+1)), dtype=np.float64)
# One-hot encode the labels
for idx, label in enumerate(train_y):
    aux[idx, int(label)] = 1
train_y = aux

aux = np.zeros((test_y.shape[0], int(np.max(test_y)+1)), dtype=np.float64)
# One-hot encode the labels
for idx, label in enumerate(test_y):
    aux[idx, int(label)] = 1
test_y = aux


# Transpose and scale training and testing data
train_x = train_x.T
# scaler = preprocessing.StandardScaler().fit(train_x)
# train_x = scaler.transform(train_x)

test_x = test_x.T
# test_x = scaler.transform(test_x)
################################################################################

########################### Model Creation #####################################
model = keras.Sequential(
    [
        keras.layers.Dense(train_x.shape[1], activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(5, activation="softmax")
    ]
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
################################################################################

######################## Model Training and Saving #############################
history = model.fit(
    train_x,
    train_y,
    epochs=20,
    validation_data=(test_x, test_y)
    )
model.save("trainedClassificationNN")
################################################################################

############################## Results Inspection ##############################
predictions = model.predict(train_x)
pred_y = (predictions > 0.5)

matrix = metrics.confusion_matrix(train_y.argmax(axis=1), pred_y.argmax(axis=1), normalize="true")
matrix = np.round(matrix*100,2)
print(matrix)

plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

plt.plot(history.history["val_loss"], color="orange")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()
################################################################################