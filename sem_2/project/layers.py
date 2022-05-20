from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

input_shape = (1, 28, 28, 1)
x = tf.random.normal(input_shape)
layer_conv2D = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))

x_ = layer_conv2D(x)

print(x_.numpy().shape)



