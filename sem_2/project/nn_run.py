import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import CSVLogger

import sklearn.metrics as metrics

from emnist import extract_training_samples, list_datasets

list_of_datasets = list_datasets()
print(list_of_datasets)

images, labels = extract_training_samples('letters')

HEIGHT = images.shape[1]
WIDTH = images.shape[2]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# for i in range(100, 109):
#     plt.subplot(330 + (i+1))
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#
# plt.show()

num_classes = np.unique(y_train).shape[0] + 1

train_y = np_utils.to_categorical(y_train, num_classes)
test_y = np_utils.to_categorical(y_test, num_classes)

X_train = X_train.reshape(-1, HEIGHT, WIDTH, 1)
X_test = X_test.reshape(-1, HEIGHT, WIDTH, 1)

# partition to train and val
X_train, val_x, train_y, val_y = train_test_split(X_train, train_y, test_size=0.10, random_state=7)

model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu',
                 input_shape=(HEIGHT, WIDTH, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

csv_logger = CSVLogger('log.csv', append=True, separator=';')
history = model.fit(X_train, train_y, epochs=10, batch_size=512, verbose=1,
                    validation_data=(val_x, val_y), callbacks=[csv_logger])


# plot accuracy and loss
def plotgraph(epochs, acc, val_acc, savefig):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(savefig)
    plt.show()

# ['accuracy', 'loss', 'val_accuracy', 'val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plotgraph(epochs, acc, val_acc, "Validation accuracy")
plotgraph(epochs, loss, val_loss, "Validation loss")

score = model.evaluate(X_test, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = metrics.confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
