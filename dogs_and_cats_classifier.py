
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