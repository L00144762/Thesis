# set the matplotlib backend so figures can be saved in the background
import matplotlib

# matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.cnn_build import CNN_NET
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
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

# x = CNN_NET
# x = x.build(32,32, 1, cls= 4)
# print(x.summary())
print("----- IMAGE PROCESSING: PLEASE STANDBY------")
data = []
labels = []
files = []

# declaring the paths for imags, val/training plot, label bin. and cnn_model
# these will then be called using command line arguments for practicality
# will need to parse the arguments for each so they can be called

#path = "C:/Users/rcurran.GARTANTECH/Desktop/Aten/keras-tutorial/animals"
path = 'C:/Users/Richard/Desktop/dissertation/images1'
#path = "C:/Users/Richard/Desktop/keras-tutorial/animals"
fig_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/train_val_plots_2"
model_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_bins/CNN_model_2.model"
lb_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/label_bin_2.pickle"

for folder in os.listdir(path):
    #print(folder)
    for file in os.listdir(path + '/' + folder):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'): #causing trouble
                                                                                        #unsure why tho
            files.append(file)
            img = path + '/' + folder + '/' + file
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (32,32))
            data.append(img)
            labels.append()


# randomize the data, labels and folders
# the folders variable is just a sanity check for labels and images
data, labels, files = shuffle(data, labels, files, random_state = 1548)

# print(files[589])
# print(data[589])
# print(labels[589])

data = np.array(data, dtype='float')
data = data/255.0
data = data.reshape(len(data),32,32,1) #reshaping the data to fit the model
                                       # there is no RGB dimension and the data needs to be
                                       # reshaped and a new dimension added
print(data.shape)
labels = np.array(labels)


# print(files[589])
#print(data[589])
# print(labels[589])

trainX, testX, trainY, testY = train_test_split(data, labels, test_size= 0.25, random_state= 25)

#print(trainY[2])
# making binary matrices for each label
# fit_transform simply means doing some calc on the train labels and transforming them to binary form
# the 'fit' is saved internally and applied to testY labels; prediction of labels learned during training
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print(trainX.shape)
print(trainY.shape)
print(len(trainX))
print(lb.classes_)
# augment the data so the nework generalizes better and helps avoid overfitting the model to training data
# fill mode means points that are filled to the poitns nearet the boundary

#
#trainX = trainY.reshape()

aug = ImageDataGenerator(rotation_range= 20, width_shift_range= 0.1,
                         height_shift_range= 0.1, zoom_range= 0.2,
                         horizontal_flip= True, fill_mode= "nearest")

# initialize the CNN
#lb.classes_ gives the length of classes that were extracted during
# 'binarizing' the labels i.e. the folder names
# keeping this as an adjustable variable means more or fewer classes
# can be trained simply by adding them to the main data directory

model = CNN_NET.build(w = 32, h = 32, d =1, cls= len(lb.classes_))
EPOCHS = 50
BS = 32
INIT_LR = 0.001

#initialize the model and optimizer
print("-----NETWORK IS TRAINING------")
opt = SGD(lr = INIT_LR, decay=INIT_LR/ EPOCHS) #step decay
model.compile(loss= "categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

H = model.fit_generator(aug.flow(trainX, trainY, batch_size= BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs= EPOCHS) # batch size is needed for augmented
                                        # training data. the training data
                                        # is passed into the network in
                                        # in augmented batches

print("-----VALIDATING NETWORK PLEASE STANDBY------ ")
predictions = model.predict(testX, batch_size= 32)
print(classification_report(testY.argmax(axis = 1),
                            predictions.argmax(axis=1),
                            target_names= lb.classes_))
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label = "training loss")
plt.plot(N, H.history["val_loss"], label = "validation loss")
plt.plot(N, H.history["accuracy"], label = "train accuracy")
plt.plot(N, H.history["val_accuracy"], label = "validation accuracy")
plt.title("Trainig Loss and Accuracy (CNN)")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(fig_save)

## save the model to use on unseen images
print("[INFO] saving network and label binarizer on memory..")
model.save(model_save)
f = open(lb_save, "wb")
f.write(pickle.dumps(lb))
f.close()

# training cnn_v1 with black and white images and will test on black and white images.
# then if tests good, retrain on coloured images and of smaller dataset (augmenting after BS size = 32)
## cnn_v1 has 6 convolutional layers
# mention kernel size, filter size, max pooling over convolution with larger kernel to reduce spatial dimension
# talk about padding and why it's set to same as I want the inout vol. size