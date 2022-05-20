import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
from keras.callbacks import CSVLogger

from sklearn.neural_network import MLPClassifier
import pickle
import sklearn.metrics as metrics
from my_utils import pad_and_normalize
from emnist import extract_training_samples, list_datasets

with open('letters.pickle', 'rb') as f:
    images, labels = pickle.load(f)

HEIGHT = images.shape[1]
WIDTH = images.shape[2]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

num_classes = np.unique(y_train).shape[0] + 1

train_y = np_utils.to_categorical(y_train, num_classes)
test_y = np_utils.to_categorical(y_test, num_classes)

X_train = X_train.reshape(-1, HEIGHT, WIDTH, 1)
X_test = X_test.reshape(-1, HEIGHT, WIDTH, 1)

# partition to train and val
X_train, val_x, train_y, val_y = train_test_split(X_train, train_y, test_size=0.10, random_state=7)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

print(X_train.shape, y_train.shape)

clf.fit(X_train, y_train)
