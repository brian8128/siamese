'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_siamese_graph.py
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
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

from settings import NB_EPOCH_CONV
from settings import NB_CONV_FILTERS

from src import data_reader

def euclidean_distance(inputs):
    assert len(inputs) == 2, ('Euclidean distance needs '
                              '2 inputs, %d given' % len(inputs))
    u, v = inputs.values()
    return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))


def contrastive_loss(y, d):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We want y and d to be different.
    Loss is 0 if y = 1 and d = 0
    Loss is 1 if y=d=1 or y=d=0
    '''
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))


def create_pairs(x, y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''

    pairs = []
    labels = []

    for i in range(5):
        perm = np.random.permutation((range(x.shape[0])))
        x = x[perm]
        y = y[perm]

        label_set = np.unique(y)
        digit_indices = {i:np.where(y == i)[0] for i in label_set}

        n = min([len(digit_indices[d]) for d in label_set]) - 1
        for d in label_set:
            for i in range(n):
                # Add a pair where the digits are the same
                z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
                pairs += [[x[z1], x[z2]]]

                # Add a pair where the digits are different
                dn = np.random.choice(label_set)
                while dn == d:
                    dn = np.random.choice(label_set)
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]

                # Add the labels for both pairs we just added
                labels += [1, 0]

    return np.array(pairs), np.array(labels)


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


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # Same is labeled 1, different is labeled 0
    # Due to the contrastive loss function we want the prediction to be 0 when
    # the label is 1.

    # Correct predictions that the two observations come from different classes
    p = (1 - labels[predictions.ravel() >= 0.3]).mean()
    print("Correctly labled imposter pairs:{}".format(p))

    # Correctly predict that the two observations come from the same class
    return labels[predictions.ravel() < 0.3].mean()


# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = data_reader.get_timeseries_data()
X_test, y_test = data_reader.get_timeseries_data('train')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

input_shape = (9, 128, 1)
nb_epoch = NB_EPOCH_CONV

# create training+test positive and negative pairs

tr_pairs, tr_y = create_pairs(X_train, y_train)
te_pairs, te_y = create_pairs(X_test, y_test)

# network definition
base_network = create_base_network(input_shape)

g = Graph()
g.add_input(name='input_a', input_shape=input_shape)
g.add_input(name='input_b', input_shape=input_shape)
g.add_shared_node(base_network, name='shared', inputs=['input_a', 'input_b'],
                  merge_mode='join')
g.add_node(Lambda(euclidean_distance), name='d', input='shared')
g.add_output(name='output', input='d')

# train
opt = RMSprop()
g.compile(loss={'output': contrastive_loss}, optimizer=opt)
g.fit({'input_a': tr_pairs[:, 0], 'input_b': tr_pairs[:, 1], 'output': tr_y},
      validation_data={'input_a': te_pairs[:, 0], 'input_b': te_pairs[:, 1], 'output': te_y},
      batch_size=128,
      nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = g.predict({'input_a': tr_pairs[:, 0], 'input_b': tr_pairs[:, 1]})['output']
tr_acc = compute_accuracy(pred, tr_y)
pred = g.predict({'input_a': te_pairs[:, 0], 'input_b': te_pairs[:, 1]})['output']
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
