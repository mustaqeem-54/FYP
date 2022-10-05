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
data = numpy.zeros((32, 320, 160, 3), dtype=np.float16)
labels = numpy.zeros((32, 2))
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
for j in range(0, 5):
    for i in range(0, 2048):

        path2 = path + 'output/'
        # path2 = path + 'Test_images/'

        path3 = path2 + new_row[(j*2048)+i]
        print(path3)