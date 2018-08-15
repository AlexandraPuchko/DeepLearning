import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# set seed for results repetition
numpy.random.seed(42)

# load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# data normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vector to categories
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# set up a model
model = Sequential()

# 1st convolutional layer
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, 32, 32),
                        activation='relu'))
# 2nd convolutional layer
model.add(Convolution2D(32, 3, 3,activation='relu'))

# Subsampling layer (max pooling)
model.add(MaxPooling2D(pool_size=(2, 2)))

# regularization layer
model.add(Dropout(0.25))

# convert 2D to Flat
model.add(Flatten())

# add fully connected layer
model.add(Dense(512, activation='relu'))

# add regularization layer
model.add(Dropout(0.5))

# output layer
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])
# train model
mode.fit(X_train, Y_train,
        batch_size=32,
        nb_epoch=25,
        validation_split=0.1,
        shuffle=True)

# evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy score on test data: %.2f%%" % (scores[1]*100))
