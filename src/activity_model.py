import numpy as np
np.random.seed(1)  # for reproducibility

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2
from src import data_source
from keras.optimizers import RMSprop
# model reconstruction from JSON:
from keras.models import model_from_json
import os

from settings import PROJECT_HOME, DROPOUT, DROPOUT_FRACTION, CONVO_DROPOUT_FRACTION, \
    NB_EPOCH, LEARNING_RATE, INPUT_SHAPE, L1_FILTERS, L2_FILTERS

ARCHITECTURE_FILE = "{}/saved_models/archetecture.json".format(PROJECT_HOME)
WEIGHTS_FILE = '{}/saved_models/model_weights.h5'.format(PROJECT_HOME)


def create_base_network(input_shape):
    """
    Base network to be shared (eq. to feature extraction).
    This is shared among the 'siamese' embedding as well as the
    more traditional classification problem
    """
    seq = Sequential()
    seq.add(Convolution2D(L1_FILTERS, 8, 1,
                          border_mode='valid',
                          activation='relu',
                          input_shape=input_shape,
                          name="input"
                          ))
    seq.add(MaxPooling2D(pool_size=(2, 1)))
    seq.add(Convolution2D(L2_FILTERS, 4, 1,
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


def create():
    # network definition
    model = create_base_network(INPUT_SHAPE)
    # output layer
    model.add(Dense(3, activation='softmax'))

    return model

def train():
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

    model = create(input_shape=input_shape)

    opt = RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=128,
              nb_epoch=nb_epoch)

    # save as JSON
    json_string = model.to_json()
    with open(ARCHITECTURE_FILE, "w") as file:
        file.write(json_string)
    model.save_weights(WEIGHTS_FILE)

    return model

def maybe_train():
    """
    Tries to load a trained model from disk but trains a new one
    if we can't load it from disk.
    :return: trained model
    """

    try:
        if os.getenv('FORCE_TRAIN', "FALSE").lower() == 'true':
            # Skip down to the except block
            # Ugly - how to make this better?
            raise Exception()

        print("attempting to load model from disk")
        with open(ARCHITECTURE_FILE, "r") as file:
            json_string = file.read()
        model = model_from_json(json_string)
        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.load_weights(WEIGHTS_FILE)

        print("Successfully loaded model from disk. No training needed.")
        return model

    except Exception:
        print("Unable to load model from disk. Training a new one")
        return train()


