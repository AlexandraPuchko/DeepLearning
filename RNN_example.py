import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# set seed
np.random.seed(42)

# max number of words (based on words frequence)
max_features = 5000

# load data
(X_train, y_train), (X_test, y_test) = imdb.load(nb_words=max_features)

# max length of words in a text exerpt
maxlen = 80
X_train = sequence.pad_sequence(X_train, maxlen=maxlen)
X_test = sequence.pad_sequence(X_test,maxlen=maxlen)

# set up model
model = Sequential()

# 1st layer for words vector representation
model.add(Embedding(max_features, 32, dropout=0.2))
# LSTM layer
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
# fully connnected layer
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
# train model
model.fit(X_train, y_train,
            batch_size=64,
            np_epoch=7,
            validation_data=(X_test, y_test)
            verbose=1)

# model evalutaion
scores = model.evaluate(X_test, y_test, batch_size=64)
print("Accuracy on test dataset: %.2f%%" % (scores[1]*100))
