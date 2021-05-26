import numpy as np

train_images = []
test_images = []
train_labels = []
test_labels = []

train_idx = 0
test_idx = 0
idx = 0
with open("ExpressionClassification/fer2013.csv") as f:
    next(f)
    for line in f:
        emotion, pixels, set = line.split(',')

        img = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48,48)
        if "Training" in set:
            train_images.append(img)
            train_labels.append(emotion)
            train_idx += 1
        elif "PrivateTest" in set or "PublicTest" in set:
            test_images.append(img)
            test_labels.append(emotion)
            test_idx += 1
        idx += 1
    
    train_images_arr = np.empty((48,48,len(train_labels)), dtype=np.uint8)
    test_images_arr = np.empty((48,48,len(test_labels)), dtype=np.uint8)
    train_labels_arr = np.empty((len(train_labels)))
    test_labels_arr = np.empty((len(test_labels)))

    for i in range(len(train_labels)):
        train_images_arr[:,:,i] = train_images[i]
        train_labels_arr[i] = train_labels[i]
    
    for i in range(len(test_labels)):
        test_images_arr[:,:,i] = test_images[i]
        test_labels_arr[i] = test_labels[i]

    np.save('train_images.npy', train_images_arr)
    np.save('test_images.npy', test_images_arr)
    np.save('train_labels.npy', train_labels_arr)
    np.save('test_labels.npy', test_labels_arr)
