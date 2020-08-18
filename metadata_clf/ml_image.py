import cv2
import os
import numpy as np
from tensorflow import keras

# Using a Keras Model
DATADIR = "./Images_noBG"
CATEGORIES = ["Y","N"]
train_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    class_num = CATEGORIES.index(category)
    for im in os.listdir(path):
        try:
            im_array = cv2.imread(os.path.join(path,im),cv2.IMREAD_COLOR)
            im_array = np.resize(im_array, (240,320,3))
            train_data.append([im_array,class_num])
        except Exception as e:
            print(e)

X_train = []
Y_train = []
for feat, label in train_data:
    X_train.append(feat)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),input_shape=(240,320,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=5,shuffle=True)
test_loss, test_acc = model.evaluate(X_train,Y_train)
print("Tensorflow: ",test_acc)
