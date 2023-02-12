
#import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#To read files and directory structure
import os

#Display images in our training and validation data
import matplotlib.pyplot as plt

#For matrix math outside Tensorflow
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', original=_URL, extract=True)

#You can list the directories with the following terminal commands
zip_dir_base = os.path.dirname(zip_dir)
!find $zip_dir_base -type d -print

#We now assign variables with the proper file path for training and validation sets
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#Understanding our data in terms of how many elements we have in our training and validation datasets
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

num_tt = total_val + total_train

print('total training cats images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print('-----')
print('Total training images:', total_train)
print('Total validation images:', total_val)
print('-----')
print('Total size of cats and dogs dataset:', num_tt)

#Setting model parameters for convenience
'''Number of training examples to process before
updating our models variables'''
BATCH_SIZE = 100
'''Our training data consists of images with width 
of 150 pixels and a height of 150 pixels 
(150*150)'''
IMG_SHAPE = 150

#Data prep