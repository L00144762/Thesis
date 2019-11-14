# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.cnn_build import CNN_NET
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

x = CNN_NET
x = x.build(32,32,3)
print(x.summary())
print("[INFO] loading images ...")
data = []
labels = []
files = []


path = "C:/Users/rcurran.GARTANTECH/Desktop/Aten/keras-tutorial/animals"
for folder in os.listdir(path):
    #print(folder)
    for file in os.listdir(path + '/' + folder):
        if file.endswith('.jpg') or file.endswith('.JPEG') or file.endswith('.png'):
            files.append(file)
            img = path + '/' + folder + '/' + file
            img = cv2.imread(img)
            img = cv2.resize(img, (32,32))
            data.append(img)
            labels.append(folder)


# randomize the data, labels and folders
# the folders variable is just a sanity check for labels and images
data, labels, files = shuffle(data, labels, files, random_state = 1548)

# print(files[589])
# print(data[589])
# print(labels[589])

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# print(files[589])
# print(data[589])
# print(labels[589])

trainX, testX, trainY, testY = train_test_split(data, labels, test_size= 0.25, random_state= 25)
# print(trainY[2])
# making binary matrices for each label
# fit_transform simply means doing some calc on the train labels and transforming them to binary form
# the 'fit' is saved internally and applied to testY labels; prediction of labels learned during training
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# print(trainY[2])


# augment the data so the nework generalizes better and helps avoid overfitting the model to training data

aug - ImageDataGenerator(rotation_range= 20, width_shift_range= 0.1,
                         height_shift_range= 0.1, shear_range= 0.2, zoom_range= 0.2,
                         horizontal_flip= True, fill_mode= "nearest")




