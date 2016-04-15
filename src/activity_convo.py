'''
Trains a convolutional neural network to predict whether someone is walking upstairs,
downstairs or walking not on stairs using normal categorical cross entropy.
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1)  # for reproducibility

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Lambda, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2, l1l2
from keras.layers import Convolution2D, MaxPooling2D

from sklearn.preprocessing import OneHotEncoder

from settings import NB_EPOCH_CONV
from settings import NB_CONV_FILTERS
from src import data_reader


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(NB_CONV_FILTERS, 10, 1,
                            border_mode='valid',
                            activation='relu',
                            input_shape=input_shape
                          ))
    seq.add(MaxPooling2D(pool_size=(3, 1)))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu',
                  ))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu',
                  W_regularizer=l2(0.01),
                  b_regularizer=l2(0.01)
                  ))
    seq.add(Dropout(0.1))
    seq.add(Dense(64, activation='relu',
                  W_regularizer=l2(0.01),
                  b_regularizer=l2(0.01)
                  ))
    return seq

# Scaled and shuffled data
X_train, subject_train, activity_train = data_reader.get_timeseries_data('train')
X_test, subject_test, activity_test = data_reader.get_timeseries_data('test')

encoder = OneHotEncoder()
activity_train_one_hot = encoder.fit_transform(activity_train).todense()
activity_test_one_hot = encoder.transform(activity_test).todense()

y_train = activity_train_one_hot
y_test = activity_test_one_hot

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

input_shape = (9, 128, 1)
nb_epoch = NB_EPOCH_CONV

# network definition
model = create_base_network(input_shape)
# output layer
model.add(Dense(3, activation='softmax'))

# train
opt = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=opt)
model.fit(X_train, y_train,
      validation_data=(X_test, y_test),
      batch_size=128,
      nb_epoch=nb_epoch)

pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])