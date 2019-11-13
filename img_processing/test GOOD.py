
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
from sklearn.preprocessing import LabelBinarizer

import time
import tensorflow as tf
#
path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/animals'
#path = 'C:/Users/Richard/Desktop/dissertation/test'
# # list of folders in image directory animals
image_folders = [f for f in listdir(path) if isdir(join(path, f))]
print(image_folders)
# # creates a dictionary of image folder names and assigns a label to each folder
image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
print(image_labels)

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

#l = 0

def im_proc_data():
    #using list comprehenstion to avoic manually typing out each class
    # this means that if this was to be used to processes images of different class
    # the path is the only thing that would need to be altered
    image_folders = [f for f in listdir(path) if isdir(join(path, f))]
    image_labels = {image_folders[i]: i for i in range(0, len(image_folders))}

    images = []
    labels = []

    for folder in listdir(path):
        files.append(folder)
        for jpeg in listdir(path + "/" + folder):
            if jpeg.endswith("jpeg") or jpeg.endswith("jpg") or jpeg.endswith("PNG"):
                jpeg = path + "/" + folder + "/" + jpeg
                jpeg = cv2.imread(jpeg, 0)
                jpeg = cv2.resize(jpeg, (32, 32))
                #jpeg = cv2.GaussianBlur(jpeg, (5,5), 0)
                #jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2GRAY)
                images.append(jpeg)
                labels.append(image_labels[folder]) # assigning label to each class



    images, labels = shuffle(images, labels, random_state = 12551)
    return(images, labels)

images, labels = im_proc_data()


# sample of the images and their label
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel(image_folders[labels[i]])
#plt.show()

images = np.array(images, dtype = 'float32')

#normalizing the images
images = images/255.0
labels = np.array(labels, dtype = 'int32')
np.save('images', images)
np.save('labels', labels)
print(images.shape)
print(labels.shape)

#reshaping the label array as a 1D array so it's compatable with model
labels = labels.reshape(-1,1)
print(labels)
print(labels.shape)

## splitting the images and labels into train, validation and test sets

# train = 75%, Validation = 15%, test = 10%
img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=.25)

img_val, img_test, label_val, label_test = train_test_split(img_test, label_test, test_size=.4)

print('Traing shape :\n')
print(img_train.shape)
print(label_train.shape)
print('validation shape: \n')
print(img_val.shape)
print(label_val.shape)
print('test shape: \n')
print(img_test.shape)
print(label_test.shape)

np.savez('images.npz', train_img = img_train, val_img = img_val, test_img = img_test)
np.savez('labels.npz', train_label = label_train, val_label = label_val, test_label = label_test)




# normalise the image data to take a value of between 0 and 1; makes computation easier

img_train, img_val, img_test = img_train/255.0, img_val/255.0, img_test/255.0

print(img_train[67], img_val[67], img_test[67])

#example of an images and their labels
# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(images[45])
# axarr[0,0].set_title(labels[45])
# axarr[0,1].imshow(images[270])
# axarr[0,1].set_title(labels[270])
# axarr[1,0].imshow(images[10])
# axarr[1,0].set_title(labels[10])
# axarr[1,1].imshow(images[170])
# axarr[1,1].set_title(labels[170])
# #plt.show()


## when splitting into train/test: must divide by 255 to normalize
# the image array
#
# img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=0.20, random_state= 155)
# img_train, img_test = img_train/255.0, img_test/255.0  ## normlaizing he pixel values to have a value between o and 1; easier computation
# print(img_train.shape)
# print(label_train.shape)
# print(img_test.shape)
# print(label_test.shape)
# #
# # x = pd.DataFrame({"images": im_files, "label": labels})
# # #print(x.loc[0, 'images'])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape = (100,100)),
#     tf.keras.layers.Dense(128, activation = 'relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation = 'softmax')
# ])
# model.compile(optimizer = 'adam',
#               loss = 'sparse_categorical_crossentropy',
#               metrics = ['accuracy'])
# model.fit(img_train, label_train, epochs = 5)
#
# #model.evaluate(img_test, label_test, verbose = 2) #models training accuracy is roughly 45%. this is a smalle network with
#                                                    # with very few layers which is why it's so poor




