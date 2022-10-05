import keras.backend
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import csv

model_runer = 1

cur_path = os.getcwd()
# Retrieving the images and their labels
path = 'C:/FYP/Dataset/Images/New_data/'
# path = 'C:/FYP/Dataset/Images/Temp/'

path1 = os.path.join(path + 'dataset.csv')
# path1 = os.path.join(path + 'Test_images.csv')
temp = open(path1)
file = csv.reader(temp)
new_row = []
for row in file:
    new_row.append(row[0])
for j in range(0, 10):
    data = numpy.zeros((1024, 320, 160, 3), dtype=np.float16)
    labels = numpy.zeros((1024, 2))
    for i in range(0, 1024):

        path2 = path + 'output/'
        # path2 = path + 'Test_images/'

        path3 = path2 + new_row[(j*1024)+i]
        try:
            image = cv2.imread(path3, -1)
            # resize image
            image1 = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
            # print(np.shape(image1))
            image2 = np.array(image1, dtype=np.float16)
            image2 = image2 / 255

            # sim = Image.from array(image)
            data[i, :, :, :] = image2
            labels[i, 0] = (float(row[1])) * 50
            labels[i, 1] = (float(row[2])) * 10


        except:
            print("Error in loading image!")
    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    # Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # Converting the labels into one hot encoding
    # y_train = to_categorical(y_train, 15)
    # y_test = to_categorical(y_test, 15)

    # Building the model
    if model_runer == 1:
        model = Sequential()

        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=X_train.shape[1:]))
        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2)))
        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(Conv2D(filters=64, kernel_size=(5, 5)))
        model.add(Conv2D(filters=128, kernel_size=(5, 5)))

        model.add(Dropout(rate=0.5))

        model.add(Flatten())

        model.add(Dense(100, activation='elu'))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(2, activation='linear'))
        model_runer = 0
    if model_runer == 0:
        model = load_model('C:/FYP/my_model12.h5')
    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 50
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("my_model12.h5")
    keras.backend.clear_session()
    del model


# plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# testing accuracy on test dataset

from sklearn.metrics import accuracy_score

'''
y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = cv2.resize(image, (160, 320, 3), interpolation=cv2.INTER_AREA)
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

# Accuracy with the test data
from sklearn.metrics import accuracy_score

print(accuracy_score(labels, pred))
'''
