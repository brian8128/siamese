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
from keras.layers.core import Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2, l1l2
from keras.layers.normalization import BatchNormalization

from src import data_reader

def euclidean_distance(inputs):
    assert len(inputs) == 2, ('Euclidean distance needs '
                              '2 inputs, %d given' % len(inputs))
    u, v = inputs.values()
    return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))


def contrastive_loss(y, d):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))


def create_pairs(x, label_set, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
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


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(256, input_shape=(input_dim,), activation='relu',
                  W_regularizer=l2(0.01)
                  ))
    seq.add(Dropout(0.1))
    seq.add(Dense(256, activation='relu',
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
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = data_reader.get_data()
X_test, y_test = data_reader.get_data(train=False)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

m = np.amax(X_train, axis=0)

X_train /= m
X_test /= m
input_dim = 561
nb_epoch = 20

# create training+test positive and negative pairs
digit_indices = {i:np.where(y_train == i)[0] for i in np.unique(y_train)}
tr_pairs, tr_y = create_pairs(X_train, np.unique(y_train), digit_indices)

digit_indices = {i:np.where(y_test == i)[0] for i in np.unique(y_test)}
te_pairs, te_y = create_pairs(X_test, np.unique(y_test), digit_indices)

# network definition
base_network = create_base_network(input_dim)

g = Graph()
g.add_input(name='input_a', input_shape=(input_dim,))
g.add_input(name='input_b', input_shape=(input_dim,))
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