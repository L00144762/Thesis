import shutil
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join,isdir
from sklearn.utils import shuffle
import numpy as np

# from skimage import io, transform
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# # creating a second directory for grayscaled images
# # def copy_images():
# #     shutil.copytree('C:/Users/Richard/Desktop/dissertation/images',
# #                     'C:/Users/Richard/Desktop/dissertation/images_gray')
# #
# # copy_images()
# #
# # # grayscaling all the images in each class using open cv
# # def gray(a):
# #     path = ('C:/Users/Richard/Desktop/dissertation/images_gray/{}'.format(a))
# #
# #     files = [f for f in listdir(path) if isfile(join(path, f))]  # list of filenames in image directory
# #
# #     for image in files:
# #         img = cv2.imread(os.path.join(path, image))
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #         dstPath = join(path, image)
# #         cv2.imwrite(dstPath, gray)
# #
# #     for fil in glob.glob("*.jpeg"):
# #         image = cv2.imread(fil)
# #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to greyscale
# #         cv2.imwrite(os.path.join(dstPath, fil), gray_image)
# #
# # gray('dog')
# # gray('cat')
# # # gray('sheep')
# # # gray('cow')


path = 'C:/Users/Richard/Desktop/dissertation/test'
#destpath = ('C:/Users/Richard/Desktop/dissertation/images_gray/{}'.format(a))

image_folders = [f for f in listdir(path) if isdir(join(path, f))]  # list of folders in image directory animals

# creates a dictionary of image folder names and assigns a label to each folder
image_labels = {image_folders[i]: i for i in range(0,len(image_folders))}
print(image_labels)
#print(image_labels)

def load_data(directory):
    output = []
    images = []
    labels = []
    file_names = []
    for folder in listdir(directory):
        curr_label = image_labels[folder]
        print(curr_label)
        for file in listdir(directory + "/" + folder):
            img_path = directory + "/" + folder + "/" + file
            curr_img = cv2.imread(img_path)
            curr_img = cv2.resize(curr_img, (150,150))
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            images.append(curr_img)
            labels.append(curr_label)
            file_names.append(file)
    images, labels, file_names = shuffle(images, labels, file_names, random_state=817328462)
    print(images,labels)### Shuffle the data !!!
    images = np.array(images, dtype = 'float32') ### image matrices held as floats
    labels = np.array(labels, dtype = 'int32')   ### corresponding labels held as integers

    return images, labels, file_names
images, labels, file_names = load_data(path)



# x_d = images
# y_d = labels
# #
# print(x_d.shape)
# print(y_d.shape)

#