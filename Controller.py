import cv2
import numpy
import csv
import os
import matplotlib.pyplot as plt
from keras.models import load_model

cur_path = os.getcwd()
# Retrieving the images and their labels
path = 'C:/FYP/Dataset/Images/New_data/'
# path = 'C:/FYP/Dataset/Images/Temp/'

path1 = os.path.join(path + 'dataset.csv')
# path1 = os.path.join(path + 'Test_images.csv')
sp = []
an = []
var = 0
model = load_model('C:/FYP/my_model.h5')
with open(path1) as file:
    reader = csv.reader(file)
    for row in reader:
        path2 = path + 'output/'
        # path2 = path + 'Test_images/'

        path3 = path2 + row[0]
        image = cv2.imread(path3, -1)
        image = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        image = image / 255
        pred = model.predict(image)
        pred = pred / 1000
        #print(pred)
        a = (float(row[1])) * 50
        b = (float(row[2])) * 10
        c = pred[0]
        d = c[0]
        e = c[1] * -1
        f = a - (d+35)
        g = b - (e+22)
        sp.append(f)
        an.append(g)

        var = var + 1
        if var % 1000 == 0:
            print(var)
            break
        # print(f, g)
x = numpy.arange(1, numpy.shape(sp)[0] + 1)
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, sp, color="green")
plt.show()
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, an, color="green")
plt.show()
