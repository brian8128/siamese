from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2

from settings import DROPOUT, DROPOUT_FRACTION, CONVO_DROPOUT_FRACTION


def create_base_network(input_shape):
    '''
    Base network to be shared (eq. to feature extraction).
    This is shared among the 'siamese' embedding as well as the
    more traditional classification problem
    '''
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
