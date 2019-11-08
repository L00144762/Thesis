
import shutil
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join,isdir
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
import time
#
# path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/test'
path = 'C:/Users/Richard/Desktop/dissertation/test'
# # list of folders in image directory animals
image_folders = [f for f in listdir(path) if isdir(join(path, f))]
# # creates a dictionary of image folder names and assigns a label to each folder
image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
print(image_labels)

images = []
labels = []
#l = 0
for folder in listdir(path):
    label = listdir()
    for jpeg in listdir(path + "/" + folder):
        if jpeg.endswith("jpeg") or jpeg.endswith("jpg") or jpeg.endswith("PNG"):
          jpeg = path + "/" + folder + "/" + jpeg
          images.append(jpeg)
          labels.append(image_labels[folder])


          # assings a unique label to each folder(that's why it's outside the if loop
images, labels = shuffle(images, labels, random_state = 123) # random state initially: for testing the image processing
                                                             # will remove when training begins to ensure
                                                             # the data is reshuffled each time and reduce
                                                              #  bias in learning
print(images[4])
print(labels[4])

# df_images_labels = pd.DataFrame({'Images':images, 'Labels':labels})
# # df_images_labels.to_csv('test.csv',index=False)

