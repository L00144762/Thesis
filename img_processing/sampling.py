# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from cnn_pyimagesearch import CNN_NET
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
x = x.build(32,32,1)
print(x.summary())

path = "C:/Users/rcurran.GARTANTECH/Desktop/Aten/keras-tutorial/animals"
for file in os.listdir(path):
    print(file)