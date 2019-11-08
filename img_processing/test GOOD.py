
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
path = 'C:/Users/Richard/Desktop/dissertation/test'
#path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/test'
# img = (path + "/" + "cat" + "/" + "9.jpeg")
# img2 = (path + "/" + "cat" + "/" + "9.jpeg")
# img2 = cv2.imread(img2,0 )
# img = cv2.imread(img, 0 )
# img = cv2.resize(img, (244,244))
# cv2.imshow("image", img)
# cv2.imshow("image2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# first way of doing it
images = []
labels = []
#l = 0
for folder in listdir(path):
    label = listdir()
    for jpeg in listdir(path + "/" + folder):
        if jpeg.endswith("jpeg") or jpeg.endswith("jpg") or jpeg.endswith("PNG"):
          jpeg = path + "/" + folder + "/" + jpeg
          jpeg = cv2.imread(jpeg, 0)
          jpeg = cv2.resize(jpeg, (244, 244))
          #jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2GRAY)
          images.append(jpeg)
        labels.append(image_labels[folder]) # assining label to each class
        #label.append(l)

    #l = l + 1

          # assings a unique label to each folder(that's why it's outside the if loop
images, labels = shuffle(images, labels, random_state = 123) # random state initially: for testing the image processing
                                                             # will remove when training begins to ensure
                                                             # the data is reshuffled each time and reduce
                                                              #  bias in learning


images = np.array(images, dtype= 'float32')
labels = np.array(labels, dtype= 'int32')
#print(images)
#print(labels)

### second way of doing it
# images = []
# labels = []
#l = 0
# def gray_resize():
#     for folder in listdir(path):
#         label = listdir()
#         for jpeg in listdir(path + "/" + folder):
#             if jpeg.endswith("jpeg") or jpeg.endswith("jpg") or jpeg.endswith("PNG"):
#                 jpeg = path + "/" + folder + "/" + jpeg
#                 jpeg = cv2.imread(jpeg, 0)
#                 jpeg = cv2.resize(jpeg, (244, 244))
#
#                 images.append(jpeg)
#                 gray_resize.images = images
#                 gray_resize.labels = labels.append(image_labels[folder])
#                 gray_resize.labels = labels
#                 #print(gray_resize.labels)# assining label to each class
#         #label.append()
#
#     #l = l + 1
#
#           # assings a unique label to each folder(that's why it's outside the if loop
# gray_resize()
# images, labels = shuffle(gray_resize.images, gray_resize.labels,
#                          random_state = 123)
# images = np.array(images, dtype= 'float32')
# labels = np.array(labels, dtype= 'int32')
print(images.shape)
print(labels.shape)

#example of an images and their labels
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(images[45])
axarr[0,0].set_title(labels[45])
axarr[0,1].imshow(images[270])
axarr[0,1].set_title(labels[270])
axarr[1,0].imshow(images[10])
axarr[1,0].set_title(labels[10])
axarr[1,1].imshow(images[170])
axarr[1,1].set_title(labels[170])
plt.show()


## when splitting into train/test: must divide by 255 to normalize
# the image array
img_train, label_train, img_test, label_test = train_test_split(images, labels, test_size=0.20, random_state= 25)
print(img_train.shape)
print(label_train.shape)
print(img_test.shape)
print(label_test.shape)
