# CNN build version 1.0
# SGD optimizer with LR of 0.01 and step-decay = (training set length / epochs)
# keeping it as computationally simple as possible at each convolution layer
#

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
                     input_shape= inputShape))
        model.add(Activation("relu"))
        #batchnormalization: normalizes the activation function
        # before o/p is sent to next layer:
        # stabalizes training and reduces
        # number of epohcs to train model
        model.add(BatchNormalization(axis= -1))# channel is the last dimension
        model.add(MaxPooling2D(pool_size= (2,2)))
        # dropout: disconnecting random neurons between layers
        # reduce overfitting, increase accuracy and generalisation
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), padding= "same")) ## [EDIT] maybe remove a layer to reduce training time?
        model.add(Activation("relu"))                   # might reduce accuracy, hopefully not too much
        model.add(BatchNormalization(axis = -1))
        model.add(Conv2D(128, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= -1))
        model.add(Conv2D(128, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= -1))
        model.add(MaxPooling2D(pool_size= (2,2)))
        model.add(Dropout(0.25))

        #fully connected layer
        model.add(Flatten()) # flatten the input to a 1D array
        model.add(Dense(512)) # 512 neurons in FC layer
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        # this does the classification
        model.add(Dense(cls))
        model.add(Activation("softmax"))



        # return the constructed network architecure
        return model
