# Richard Curran
# L00144762
#Initial image processing: sampling the Animals-10 dataset
# URL: https://www.kaggle.com/alessiocorrado99/animals10


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.utils import to_categorical

# convert to grayscale
# resize
# make sure there's no people in them
# import  numpy as np
# from sklearn.model_selection import train_test_split
# x = np.arange(1,21)
# x = np.array(x)
# y = np.arange(1,21)
# y = np.array(y)
# print(x,y)
#
# x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=.25)
# x_val,x_test,y_val, y_test = train_test_split(x_test,y_test, test_size=.4)
# print(x_train, y_train)
# print(x_val, y_val)
# print(x_test, y_test)

images = np.load('images.npy')
labels = np.load('labels.npy')
print(images.shape)
print(labels.shape)

images = np.load('images.npz')
labels = np.load('labels.npz')

trainX = images['train_img']
trainY = labels['train_label']
testX = images['test_img']
testY = labels['test_label']
images = np.load('images.npz')
images = np.load('labels.npz')

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print(trainX[2])
print(trainY[2])

