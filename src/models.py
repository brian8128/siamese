import numpy as np
np.random.seed(1)  # for reproducibility

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2
from src import data_source
from keras.optimizers import SGD, RMSprop, Adam

from settings import DROPOUT, DROPOUT_FRACTION, CONVO_DROPOUT_FRACTION, NB_EPOCH, LEARNING_RATE


def create_base_network(input_shape):
    """
    Base network to be shared (eq. to feature extraction).
    This is shared among the 'siamese' embedding as well as the
    more traditional classification problem
    """
    seq = Sequential()
    seq.add(Convolution2D(32, 8, 1,
                          border_mode='valid',
                          activation='relu',
                          input_shape=input_shape,
                          name="input"
                          ))
    seq.add(MaxPooling2D(pool_size=(2, 1)))
    seq.add(Convolution2D(64, 4, 1,
                          border_mode='valid',
                          activation='relu'
                          ))
    seq.add(MaxPooling2D(pool_size=(2, 1)))
    if DROPOUT:
        seq.add(Dropout(CONVO_DROPOUT_FRACTION))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu',
                  ))
    if DROPOUT:
        seq.add(Dropout(DROPOUT_FRACTION))
    seq.add(Dense(128, activation='relu',
                  W_regularizer=l2(0.01),
                  b_regularizer=l2(0.01)
                  ))
    if DROPOUT:
        seq.add(Dropout(DROPOUT_FRACTION))

    return seq


def create_activity_model(input_shape):
    # network definition
    model = create_base_network(input_shape)
    # output layer
    model.add(Dense(3, activation='softmax'))

    return model

def train_activity_model():
    """
    Constructs the activity mode, gets data from data_source, trains the model
    saves model and returns the model.
    :return: trained model
    """
    # Scaled and shuffled data
    X_train, subject_train, activity_train, _ = data_source.get_timeseries_data('train')
    X_test, subject_test, activity_test, _ = data_source.get_timeseries_data('test')

    y_train = activity_train
    y_test = activity_test

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    input_shape = (X_train.shape[1], 128, 1)
    nb_epoch = NB_EPOCH

    model = create_activity_model(input_shape=input_shape)

    opt = RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=128,
              nb_epoch=nb_epoch)

    return model

def maybe_train_activity_model():
    """
    Tries to load a trained model from disk but trains a new one
    if we can't load it from disk.
    :return: trained model
    """
    pass