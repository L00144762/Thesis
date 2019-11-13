# USAGE
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

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

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required= True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help= "path to CNN model")
ap.add_argument("-l", "--label-bin", required=True,
                help= "path to output label binarizer")
ap.add_argument("-p", "--plot", required= True,
                help= "path to output accuracy/ loss plot")

args = vars(ap.parse_args())


# initialize the data and labels
print("[INFO] loading images")
data = []
labels = []

# grab the image paths and randomly shuffle them
# paths. function saves you the bother of writing multiple for loops
# to walk through a directory and find the input data
# this is part of the imutils library
# github link for function: https://github.com/jrosebr1/imutils/blob/master/imutils/paths.py

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(25)
random.shuffle(imagePaths)

for imagePath in imagePaths:
     img = cv2.imread(imagePath)
     img = cv2.resize(img, (32,32)).flatten()
     data.append(img)
     label = imagePath.split(os.path.sep)[1]
     labels.append(label)

data = np.array(data, dtype= 'float')
data = data/255.0
labels = np.array(labels)


trainX, testX, trainY, testY = train_test_split(data, labels, test_size= 0.25, random_state= 25)

# # convert the labels from integers to vectors. this is multi class
# this is a bit better than one hot encoding as one hot encoding requires integer values
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print(trainY[3])


# define neural network using keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(len(lb.classes_), activation = 'softmax'))
model.summary()

INIT_LR = 0.01
EPOCHS =4

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
plt.savefig(args["plot"])

#save the model and label binarier to disk
# converting the model & binarizer to a structure (bytes??) that can be stored on RAM and accessed later
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb") # label binarizer is being written to file in binary mode - wb
f.write(pickle.dumps(lb)) #pickling saves python objects on memory (saves disk space)
f.close() # closing binariz\er