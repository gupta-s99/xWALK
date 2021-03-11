#xWalk Training Model 
import tensorflow as tf
import numpy as np 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
dataset_url = "https://drive.google.com/drive/u/2/folders/1cnI4IMmvBUon3n-_3j6OfAv9lXRYkZcW"
data_dir = tf.keras.utils.get_file('JPG', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)