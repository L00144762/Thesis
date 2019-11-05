import shutil
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join,isdir
from sklearn.utils import shuffle
import numpy as np

path = 'C:/Users/Richard/Desktop/dissertation/test'

# list of folders in image directory animals
image_folders = [f for f in listdir(path) if isdir(join(path, f))]

# creates a dictionary of image folder names and assigns a label to each folder
image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
#print(image_labels)

def images_path():
    images = []
    labels = []
    files = []
## main processing loop
    # walks through each sub-folder and searches for images ending with jpeg/jpg
    for sub_folder in listdir(path):
        for jpeg in listdir(path+ "/" + sub_folder):
            if jpeg.endswith('jpeg') or jpeg.endswith('jpg'):
                each_image = path + "/" + sub_folder + "/" + jpeg
                each_image= cv2.imread(each_image)
                each_image = cv2.resize(each_image, (224,224))
                each_image = cv2.cvtColor(each_image, cv2.COLOR_BGR2GRAY)
                images.append(each_image)
                labels.append(image_labels[sub_folder])
                files.append(jpeg)
                print(images, labels)

    # creating np objects of both labels and images
    #print(images, labels)
    images_path.images = np.array(images, dtype='float32')
    images_path.labels = np.array(labels, dtype='int32')
    return images, labels, files

images_path()

images = images_path.images
labels = images_path.labels
## shuffling the images and labels
## so they are not in the same order as in root directory#
# reduces bias
images, labels = shuffle(images, labels,  random_state = 1232434)

print(images.shape)
print(labels.shape)