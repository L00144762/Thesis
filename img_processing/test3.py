# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
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
import os,glob
from os import listdir,makedirs
from os.path import isfile,join,isdir
from keras.utils import to_categorical


# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#                 help= "path to CNN model")
# ap.add_argument("-l", "--label-bin", required=True,
#                 help= "path to output label binarizer")
# ap.add_argument("-p", "--plot", required= True,
#                 help= "path to output accuracy/ loss plot")
#
# args = vars(ap.parse_args())

#path = 'C:/Users/rcurran.GARTANTECH/Desktop/Aten/animals'
path = 'C:/Users/Richard/Desktop/dissertation/images'
# # list of folders in image directory animals
# image_folders = [f for f in listdir(path) if isdir(join(path, f))]
# print(image_folders)
# # # creates a dictionary of image folder names and assigns a label to each folder
# image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
# print(image_labels)

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
    files = []


    for folder in listdir(path):
        files.append(folder)
        for jpeg in listdir(path + "/" + folder):
            if jpeg.endswith("jpeg") or jpeg.endswith("jpg") or jpeg.endswith("PNG"):
                jpeg = path + "/" + folder + "/" + jpeg
                jpeg = cv2.imread(jpeg, 0)
                jpeg = cv2.resize(jpeg, (32, 32)).flatten()# image size for  simple neural network
                #jpeg = cv2.GaussianBlur(jpeg, (5,5), 0)
                #jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2GRAY)
                images.append(jpeg)
                labels.append(folder) # assigning label to each class(just the folder name)



    images, labels = shuffle(images, labels, random_state = 12551) # shuffling the data
    list_data = list(images, labels, files)
    return(images, labels, files)

images, labels, files = im_proc_data()

images = np.array(images, dtype = 'float32')
labels = np.array(labels)
# print(images.shape)
# print(labels.shape)

#normalizing the images
images = images/255.0

# np.save('images', images)
# np.save('labels', labels)
# print(images.shape)
# print(labels.shape)

#reshaping the label array as a 1D array so it's compatable with model
# labels = labels.reshape(-1,1)
# print(labels)
# print(labels.shape)

## splitting the images and labels into train, validation and test sets

# train = 75%, Validation = 15%, test = 10%
img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=.25)

img_val, img_test, label_val, label_test = train_test_split(img_test, label_test, test_size=.4)
#
# print('Traing shape :\n')
# print(img_train.shape)
# print(label_train.shape)
# print('validation shape: \n')
# print(img_val.shape)
# print(label_val.shape)
# print('test shape: \n')
# print(img_test.shape)
# print(label_test.shape)
#

trainX = img_train
trainY = label_train
testX = img_val
testY = label_val


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print(trainY.shape)
print(testY.shape)

#saving the training, validation and testing datasets as numpy objects for easy calling in a different script
np.savez('images.npz', train_img = img_train, val_img = img_val, test_img = img_test)
np.savez('labels.npz', train_label = label_train, val_label = label_val, test_label = label_test)

lb = LabelBinarizer()
label_train = lb.fit_transform(label_train)
label_test = lb.transform(label_test)
print("\n[INFO] after LabelBinarizer")
print(img_train.shape)
print(label_train.shape)

np.savez('images.npz', trainX = trainX, testX= testX)
np.savez('labels.npz', trainY = trainY, testY = testY)

# define neural network using keras
model = Sequential()
model.add(Dense(1024,input_shape= (1024,), activation='sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(len(lb.classes_), activation = 'softmax'))
model.summary()

INIT_LR = 0.01
EPOCHS = 30

# training model
print("[INFO] training network....")
opt = SGD(lr = INIT_LR)
model.compile(loss = 'categorical_crossentropy', optimizer= opt,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs= EPOCHS, batch_size=32)

# evaluate teh network
print("[INFO] evaluating the network ..")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# an array of numbers ranging of 0 to the number of epochs in steps of 1
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history['loss'], label = 'train_loss')
plt.plot(N, H.history['val_loss'], label = 'val_loss')
plt.plot(N, H.history["accuracy"], label = 'train_acc')
plt.plot(N, H.history["val_accuracy"], label = 'val_acc')
plt.title("Training Loss and Accuracy (MLP NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
#plt.savefig(args["plot"])

#save the model and label binarier to disk
# converting the model & binarizer to a structure (bytes??) that can be stored on RAM and accessed later
# print("[INFO] serializing network and label binarizer...")
# model.save(args["model"])
# f = open(args["label_bin"], "wb") # label binarizer is being written to file in binary mode - wb
# f.write(pickle.dumps(lb)) #pickling saves python objects on memory (saves disk space)
# f.close() # closing binariz\er
#
