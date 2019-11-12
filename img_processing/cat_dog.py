# Importing the required libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images/255.0, test_images/255.0
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
# print(test_images[:3])
# print(test_labels[:3])


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
#
plt.figure(figsize= (10,10))
for i in range(25):
    plt.subplot(5,5, i +1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
#plt.show()
def CNN_model():
    model =  models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

    #model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(96, activation = 'relu'))
    model.add(layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print('evev')

model = CNN_model()


history = model.fit(train_images, train_labels, epochs = 1, validation_data = (test_images, test_labels))

#plt.plot(history.history['accuracy'], label = 'accuracy')