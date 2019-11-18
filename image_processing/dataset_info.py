from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from os import  listdir
from os.path import  isdir, join
import string
import pandas as pd
from collections import Counter

#

images = np.load('np_data/images.npy')
labels = np.load('np_data/labels.npy')
#files = np.load('np_data/files.npy')
clss = np.load('np_data/clss.npy')


# one hot encoder labels and their class names in dataframe
def onehot_class():
    x = list(np.unique(labels, axis=0))
    y = list(set(clss))
    label_class = {'one-hot encoded label':x, 'Class name':y}
    label_class = pd.DataFrame(label_class)
    print(label_class)
    return label_class

onehot_class()

# plot of the number of animal images in each class
def plot_animals():
    animal = Counter(clss).keys()
    animal_count = Counter(clss).values()

    img_count = {'animal': list(animal), 'Count': list(animal_count)}
    img_count = pd.DataFrame(img_count)
    data_plot = img_count.plot(kind='bar', x='animal', y='Count')

    #plt.show()
    return img_count,  data_plot

img_count, data_plot = plot_animals()
plt.show(data_plot)



