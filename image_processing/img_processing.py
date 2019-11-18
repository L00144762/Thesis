# set the matplotlib backend so figures can be saved in the background
import matplotlib

# matplotlib.use("Agg")

# import the necessary packages
#from pyimagesearch.cnn_build import CNN_NET
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from os import  listdir
from os.path import  isdir, join
import pandas as pd
from collections import Counter


# x = CNN_NET
# x = x.build(32,32, 1, cls= 4)
# print(x.summary())
# print("-------Model summary ---------")
# CNN_NET_3layers_1DO.build(w = 32, h = 32, d = 1, cls= 3).summary()


print("----- IMAGE PROCESSING: PLEASE STANDBY------")
data = []
labels = []
files = []

# declaring the paths for imags, val/training plot, label bin. and cnn_model
# these will then be called using command line arguments for practicality
# will need to parse the arguments for each so they can be called


path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/animals'
#path = 'C:/Users/Richard/Desktop/dissertation/images1'
#path = "C:/Users/Richard/Desktop/keras-tutorial/animals"
fig_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/train_val_plots/train_val_plots_3L_1D_LR01"
model_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_bins/CNN_model_3layers_1DO_LR01.model"
lb_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/label_bin_2.pickle"

np_data_path = 'C:/Users/rcurran.GARTANTECH/PycharmProjects/Thesis/image_processing/np_data'


folders = set(listdir(path))
folders = list(folders)
# # creates a dictionary of image folder names and assigns a label to each folder
image_labels = {folders[i]: i for i in range(0, len(folders))}

## ------------------MAIN IMAGE PROCESSING------------------------------------------------------
WIDTH = 100
HEIGHT = 100
num_classes = len(listdir(path))
def img_proc():
    images = []
    labels =[]
    clss = []
    files =[]

    for folder in listdir(path):
        #clss.append(folder)
        for file in listdir(path + '/' + folder):
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img = path + '/' + folder + '/' + file
                files.append(files)
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (WIDTH, HEIGHT))
                images.append(img)
                labels.append(image_labels[folder])
                clss.append(folder)


    images = np.array(images, dtype= 'float32') / 255.0
    labels = np.array(labels, dtype= 'int32')
    #labels = to_categorical(labels, num_classes)

    np.save(os.path.join(np_data_path, 'images'), images)
    np.save(os.path.join(np_data_path, 'labels'), labels)
    np.save(os.path.join(np_data_path, 'files'), files)
    np.save(os.path.join(np_data_path, 'clss'), clss)


    return images, labels, clss, files

img_proc()


#--------- DF OF IMAGE FILES, LABELS, CLASS----------------------------------
## dataframe of images, their label and class
# dataset = {'image':list(files), 'label':list(labels), 'class':clss}
# dataset = pd.DataFrame(dataset)
# print(dataset.head(10))


# DATA PLOT: TO SEE HOW IT IS BALANCED----------------------------------------
# plot of number of images per class in dataset
# def plot_animals():
#     animal = Counter(clss).keys()
#     animal_count = Counter(clss).values()
#
#     img_count = {'animal': list(animal), 'Count': list(animal_count)}
#     img_count = pd.DataFrame(img_count)
#     data_plot = img_count.plot(kind='bar', x='animal', y='Count')
#
#     #plt.show()
#     return img_count,  data_plot
#
# img_count, data_plot = plot_animals()

#plt.show(data_plot)

#example of an images and their labels

#just do the plot before shuffling
# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(images[45])
# axarr[0,0].set_title('{},{}'.format(labels[45], clss[45]))
# axarr[0,1].imshow(images[270])
# axarr[0,1].set_title('{},{}'.format(labels[270], clss[270]))
# axarr[1,0].imshow(images[10])
# axarr[1,0].set_title('{},{}'.format(labels[10], clss[10]))
# axarr[1,1].imshow(images[170])
# axarr[1,1].set_title('{},{}'.format(labels[170], clss[170]))
# plt.show()
#
#

# images, labels, clss = shuffle(images, labels, clss)
#
# def reshaping_():
#     trainX, testX, trainY, testY = train_test_split(images,
#                                                     labels, test_size=.10)
#     trainX = trainX.reshape(len(trainX), WIDTH, HEIGHT,1)
#     testX = testX.reshape(len(testX), WIDTH, HEIGHT, 1)
#     # print(np.unique(trainY))
#     # print(np.unique(testY))
#
#     testY = to_categorical(testY, len(listdir(path)))
#     trainY = to_categorical(trainY, len(listdir(path)))
#
#     return trainX, trainY, testX, testY
#
# trainX, trainY, testX, testY = reshaping_()
#
# # print(np.unique(trainY, axis= 0))
# # print("test unique")
# # print(np.unique(testY, axis=0))
#
# cv2.imshow('{}'.format(labels[677]), trainX[677])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print(trainX.shape)
# print(testX.shape)
#
#
# model = Sequential([
#   Conv2D(8, 3, input_shape=(WIDTH, HEIGHT, 1), padding= 'same'),
#   MaxPooling2D(pool_size=2),
#   Flatten(),
#   Dense(10, activation='softmax'),
# ])
# #
# model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #
# model.fit(trainX, trainY, validation_data=(testX, testY), epochs= 15, batch_size=20)
# #
# p = model.predict(testX, batch_size= 20)
# #
# print(classification_report(testY.argmax(axis = 1),
#                             p.argmax(axis=1),
#                             target_names= listdir(path)))
# N = np.arange(0, 15)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label = "training loss")
# plt.plot(N, H.history["val_loss"], label = "validation loss")
# plt.plot(N, H.history["accuracy"], label = "train accuracy")
# plt.plot(N, H.history["val_accuracy"], label = "validation accuracy")
# plt.title("Trainig Loss and Accuracy (CNN)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()
# plt.savefig(fig_save)

# cv2.imshow('{}'.format(trainY[677]),trainX[677])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
