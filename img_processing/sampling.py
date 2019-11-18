# set the matplotlib backend so figures can be saved in the background

# import the necessary packages
#from pyimagesearch.cnn_build import CNN_NET
#

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

# x = CNN_NET
# x = x.build(32,32, 1, cls= 4)
# print(x.summary())
# print("-------Model summary ---------")
# CNN_NET_3layers.build(w = 32, h = 32, d = 1, cls= 3).summary()

print("----- IMAGE PROCESSING: PLEASE STANDBY------")
# declaring the paths for imags, val/training plot, label bin. and cnn_model
# these will then be called using command line arguments for practicality
# will need to parse the arguments for each so they can be called

#path = "C:/Users/rcurran.GARTANTECH/Desktop/Aten/keras-tutorial/animals"
path = 'C:/Users/Richard/Desktop/dissertation/images1000'
path = "C:/Users/Richard/Desktop/keras-tutorial/animals"
fig_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/train_val_plots/train_val_plots_2"
model_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_bins/CNN_model_2_2.model"
#lb_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/label_bin_2.pickle"

image_folders = [f for f in listdir(path) if isdir(join(path, f))]
# print(image_folders)

image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
#

images = []
labels = []
files = []

for folder in os.listdir(path):
    for file in os.listdir(path + '/' + folder):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'): #causing trouble
                                                                                        #unsure why tho
            files.append(file)
            img = path + '/' + folder + '/' + file
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (32, 32))
            img = img / 255.0
            images.append(img)
            labels.append(image_labels[folder])



# randomize the data, labels and folders
# the folders variable is just a sanity check for labels and images
images, labels, files = shuffle(images, labels, files, random_state = 264)

images = np.array(images, dtype='float')
#data = data/255.0
images = images.reshape(len(images),32,32,1) #reshaping the data to fit the model
                                       # there is no RGB dimension and the data needs to be
                                       # reshaped and a new dimension added


labels = np.array(labels, dtype='int32')



# lb_bin = LabelBinarizer()
# labels = lb_bin.fit_transform(labels)
# print(labels[801])
# print(labels.classes_)



trainX, testX, trainY, testY = train_test_split(images, labels,
                                                test_size= 0.20,
                                                random_state= 300)

trainY = to_categorical(trainY, len(listdir(path)))
testY = to_categorical(testY, len(listdir(path)))

print(trainX.shape)
print(testX.shape)
#
# trainX = trainX.reshape(len(trainX), 32, 32, 1)
# testX = testX.reshape(len(testX), 32, 32, 1)



#________ Training and evaluating__________

INIT_LR = 0.01
EPOCHS = 75
opt = SGD(lr=INIT_LR)


aug = ImageDataGenerator(rotation_range= 30, width_shift_range= 0.2,
                         height_shift_range= 0.2, zoom_range= 0.25,
                         horizontal_flip= True, fill_mode= "nearest")

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
from img_processing.cnn_v3 import CNN_1layer_2DO
model = CNN_1layer_2DO.cnn(w = 32, h = 32, d  = 1, cls= len(listdir(path)))

print("[INFO] training network...")

model.compile(loss="categorical_crossentropy", optimizer='Adam',
              metrics=["accuracy"])

H = model.fit(trainX, trainY,
              validation_data=(testX, testY), epochs= EPOCHS, batch_size= 16)

H = model.fit_generator(aug.flow(trainX, trainY, batch_size= 16),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // 16,
                        epochs= EPOCHS, verbose= 0)

print("[INFO] evaluating network... Lr = 0.001")
predictions = model.predict(testX, batch_size=16)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names= listdir(path)))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

