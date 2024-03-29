
# USAGE
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
#
# # set the matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")
#
# # import the necessary packages
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import SGD
# from imutils import paths
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import random
# import pickle
# import cv2
# from os import listdir,makedirs
# from os.path import isfile,join,isdir
# # path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/test_loads'
# # # path = 'C:/Users/Richard/Desktop/dissertation/test'
# # # # list of folders in image directory animals
# # image_folders = [f for f in listdir(path) if listdir(join(path, f))]
# # print(image_folders)
# # # # creates a dictionary of image folder names and assigns a label to each folder
# image_labels = {image_folders[i]: i for i in range(0, len(image_folders))}
# print(image_labels)
#
#
# # path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/test'
# # img = (path + "/" + "cat" + "/" + "9.jpeg")
# # img2 = (path + "/" + "cat" + "/" + "9.jpeg")
# # img2 = cv2.imread(img2,0 )
# # img = cv2.imread(img, 0 )
# # img = cv2.resize(img, (244,244))
# # cv2.imshow("image", img)
# # cv2.imshow("image2", img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # first way of doing it
#
# # l = 0
#
# def im_proc_data():
#     # using list comprehenstion to avoic manually typing out each class
#     # this means that if this was to be used to processes images of different class
#     # the path is the only thing that would need to be altered
#     #image_folders = [f for f in listdir(path) if isdir(join(path, f))]
#     #image_labels = {image_folders[i]: i for i in range(0, len(image_folders))}
#
#     images = []
#     labels = []
#     files = []
#
#     imagePaths = sorted(list(paths.list_images(path)))
#     for image in imagePaths:
#         img = image
#         #img = cv2.imread(image)
#         # img = cv2.resize(img, (32,32)).flatten() # don't need a flattening layer in the MLP NN
#         images.append(img)
#     for folder in listdir(path):
#         label = folder
#         labels.append(folder)
#
#     #images, labels = shuffle(images, labels, random_state=12551)  # shuffling the data
#     return (images, labels, files)
#
#
# images, labels, files = im_proc_data()
#
# print(images[3])
# print(labels[3])

# images = np.array(images, dtype='float32')
# labels = np.array(labels)
# # print(images.shape)
# # print(labels.shape)
#
# # normalizing the images
# images = images / 255.0
#
# # np.save('images', images)
# # np.save('labels', labels)
# # print(images.shape)
# # print(labels.shape)
#
# # reshaping the label array as a 1D array so it's compatable with model
# # labels = labels.reshape(-1,1)
# # print(labels)
# # print(labels.shape)
#
# ## splitting the images and labels into train, validation and test sets
#
# # train = 75%, Validation = 15%, test = 10%
# img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=.25)
#
# img_val, img_test, label_val, label_test = train_test_split(img_test, label_test, test_size=.4)
# #
# # print('Traing shape :\n')
# # print(img_train.shape)
# # print(label_train.shape)
# # print('validation shape: \n')
# # print(img_val.shape)
# # print(label_val.shape)
# # print('test shape: \n')
# # print(img_test.shape)
# # print(label_test.shape)
# #
#
# lb = LabelBinarizer()
# label_train = lb.fit_transform(label_train)
# label_val = lb.transform(label_test)
#
# # saving the training, validation and testing datasets as numpy objects for easy calling in a different script
# np.savez('images.npz', train_img=img_train, val_img=img_val, test_img=img_test)
# np.savez('labels.npz', train_label=label_train, val_label=label_val, test_label=label_test)
#
# lb = LabelBinarizer()
# label_train = lb.fit_transform(label_train)
# label_test = lb.transform(label_test)
# print("\n[INFO] after LabelBinarizer")
# print(img_train[25])
# print(label_train[25])

# x = np.array([[50, 60, 70]], dtype="int")
#
# print(x)
# print(x.argmax(axis = 0)[1])
# path = 'C:/Users/Richard/Desktop/dissertation/images/cat/4.jpeg'
# x = cv2.imread(path)
# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
# x = cv2.resize(x, (32,32))
# x = np.array(x, dtype='float')
# print(x.shape)
# print(x)
# print(x/255.0)

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CNN_NET:
    def build(w, h, d ,cls):
        model = Sequential()
        inputShape = (w, h, d) # no dimension value: grayscale image. will add when colour images are used

        #32 filters of 3*3
        model.add(Conv2D(32, (3,3), padding= "same",
                     input_shape= inputShape, activation= 'relu'))
        #model.add(Activation("relu"))
        #batchnormalization: normalizes the activation function
        # before o/p is sent to next layer:
        # stabalizes training and reduces
        # number of epohcs to train model
        model.add(BatchNormalization(axis= -1))# channel is the last dimension
        model.add(MaxPooling2D(pool_size= (2,2)))
        # dropout: disconnecting random neurons between layers
        # reduce overfitting, increase accuracy and generalisation
        model.add(Dropout(0.25))


        model.add(Conv2D(64, (3,3), padding= "same",activation= 'relu'))
        #model.add(Activation("relu"))
        model.add(BatchNormalization(axis= -1))
        model.add(MaxPooling2D(pool_size= (2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), padding= "same", activation= 'relu'))

        #fully connected layer
        model.add(Flatten()) # flatten the input to a 1D array
        model.add(Dense(128)) # 512 neurons in FC layer
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        # this does the classification
        model.add(Dense(cls,activation= 'softmax'))
        #model.add(Activation("softmax"))



        # return the constructed network architecure
        return model




class CNN_NET_3layers:
    def build(w, h, d, cls):
        model = Sequential()
        inputShape = (w, h, d)  # no dimension value: grayscale image. will add when colour images are used

        # 32 filters of 3*3
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape, activation='relu'))
        # model.add(Activation("relu"))
        # batchnormalization: normalizes the activation function
        # before o/p is sent to next layer:
        # stabalizes training and reduces
        # number of epohcs to train model
        model.add(BatchNormalization(axis=-1))  # channel is the last dimension
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # dropout: disconnecting random neurons between layers
        # reduce overfitting, increase accuracy and generalisation
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        # model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # fully connected layer
        model.add(Flatten())  # flatten the input to a 1D array
        model.add(Dense(128))  # 512 neurons in FC layer
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        # this does the classification
        model.add(Dense(cls, activation='softmax'))
        # model.add(Activation("softmax"))

        # return the constructed network architecure
        return model


## removing all dropout layers; investgate the effect on train acc.
class CNN_NET_3layers_noDO:
    def build(w, h, d, cls):
        model = Sequential()
        inputShape = (w, h, d)  # no dimension value: grayscale image. will add when colour images are used

        # 32 filters of 3*3
        model.add(Conv2D(64, (3, 3), padding="same",
                         input_shape=inputShape, activation='relu'))
        # model.add(Activation("relu"))
        # batchnormalization: normalizes the activation function
        # before o/p is sent to next layer:
        # stabalizes training and reduces
        # number of epohcs to train model
        model.add(BatchNormalization(axis=-1))  # channel is the last dimension
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # dropout: disconnecting random neurons between layers
        # reduce overfitting, increase accuracy and generalisation
        #model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding="same", activation='relu'))
        # model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        # fully connected layer
        model.add(Flatten())  # flatten the input to a 1D array
        model.add(Dense(128))  # 512 neurons in FC layer
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        #model.add(Dropout(0.5))

        # softmax classifier
        # this does the classification
        model.add(Dense(cls, activation='softmax'))
        # model.add(Activation("softmax"))

        # return the constructed network architecure
        return model


## 3 layers with one dropout layer
class CNN_NET_3layers_1DO:
    def build(w, h, d, cls):
        model = Sequential()
        inputShape = (w, h, d)  # no dimension value: grayscale image. will add when colour images are used

        # 32 filters of 3*3
        model.add(Conv2D(64, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        # batchnormalization: normalizes the activation function
        # before o/p is sent to next layer:
        # stabalizes training and reduces
        # number of epohcs to train model
        model.add(BatchNormalization(axis=-1))  # channel is the last dimension
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # dropout: disconnecting random neurons between layers
        # reduce overfitting, increase accuracy and generalisation
        #model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # fully connected layer
        model.add(Flatten())  # flatten the input to a 1D array
        model.add(Dense(512))  # 512 neurons in FC layer
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        # this does the classification
        model.add(Dense(cls, activation='softmax'))
        # model.add(Activation("softmax"))

        # return the constructed network architecure
        return model
