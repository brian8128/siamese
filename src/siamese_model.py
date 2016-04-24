'''Train a Siamese MLP on pairs of observations from the Human Activity Recognition 
Using Smartphones Data Set dataset.  It follows Hadsell-et-al.'06 [1] by computing 
the Euclidean distance on the output of the shared network and by optimizing the 
contrastive loss (see paper for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_siamese_graph_convo.py

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(1)  # for reproducibility

from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Input
from keras.callbacks import EarlyStopping

from settings import  OPTIMIZER, INPUT_SHAPE, PROJECT_HOME
from keras.models import model_from_json

from src import data_source
from sklearn.metrics import roc_auc_score

import os

ARCHITECTURE_FILE = "{}/saved_models/siamese_archetecture.json".format(PROJECT_HOME)
WEIGHTS_FILE = '{}/saved_models/siamese_model_weights.h5'.format(PROJECT_HOME)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


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


def create_base_network(
                    c1_filters,
                    c1_W_regularizer,
                    c1_b_regularizer,
                    c1_dropout,
                    c1_width,
                    c2_filters,
                    c2_W_regularizer,
                    c2_b_regularizer,
                    c2_dropout,
                    c2_width,
                    d1_size,
                    d1_W_regularizer,
                    d1_b_regularizer,
                    d1_dropout,
                    d2_size,
                    d2_W_regularizer,
                    d2_b_regularizer,
                    d2_dropout,
                    embedding_dim,
                    embedding_W_regularizer,
                    embedding_b_regularizer,
                    margin,
                    epochs,
                    learning_rate):
    """
    Take the base network and add an embedding layer
    """
    seq = Sequential()
    seq.add(Convolution2D(c1_filters, c1_width, 1,
                          border_mode='valid',
                          activation='relu',
                          input_shape=INPUT_SHAPE,
                          name="input",
                          W_regularizer=l2(c1_W_regularizer),
                          b_regularizer=l2(c1_b_regularizer),
                          ))
    seq.add(MaxPooling2D(pool_size=(3, 1)))
    seq.add(Dropout(c1_dropout))
    if c2_filters > 0:
        seq.add(Convolution2D(c2_filters, c2_width, 1,
                              border_mode='valid',
                              activation='relu',
                              W_regularizer=l2(c2_W_regularizer),
                              b_regularizer=l2(c2_b_regularizer),
                              ))
        seq.add(MaxPooling2D(pool_size=(3, 1)))
        seq.add(Dropout(c2_dropout))

    seq.add(Flatten())
    seq.add(Dense(d1_size, activation='relu',
                  W_regularizer=l2(d1_W_regularizer),
                  b_regularizer=l2(d1_b_regularizer),
                  ))
    seq.add(Dropout(d1_dropout))
    seq.add(Dense(d2_size, activation='relu',
                  W_regularizer=l2(d2_W_regularizer),
                  b_regularizer=l2(d2_b_regularizer),
                  ))
    seq.add(Dropout(d2_dropout))

    embedding = Dense(embedding_dim, activation='linear',
                      W_regularizer=l2(embedding_W_regularizer),
                      b_regularizer=l2(embedding_b_regularizer),
                      name='embedding'
                      )

    seq.add(embedding)

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


def create(c1_filters,
                    c1_W_regularizer,
                    c1_b_regularizer,
                    c1_dropout,
                    c1_width,
                    c2_filters,
                    c2_W_regularizer,
                    c2_b_regularizer,
                    c2_dropout,
                    c2_width,
                    d1_size,
                    d1_W_regularizer,
                    d1_b_regularizer,
                    d1_dropout,
                    d2_size,
                    d2_W_regularizer,
                    d2_b_regularizer,
                    d2_dropout,
                    embedding_dim,
                    embedding_W_regularizer,
                    embedding_b_regularizer,
                    margin,
                    epochs,
                    learning_rate):
    # network definition
    base_network = create_base_network(c1_filters,
                    c1_W_regularizer,
                    c1_b_regularizer,
                    c1_dropout,
                    c1_width,
                    c2_filters,
                    c2_W_regularizer,
                    c2_b_regularizer,
                    c2_dropout,
                    c2_width,
                    d1_size,
                    d1_W_regularizer,
                    d1_b_regularizer,
                    d1_dropout,
                    d2_size,
                    d2_W_regularizer,
                    d2_b_regularizer,
                    d2_dropout,
                    embedding_dim,
                    embedding_W_regularizer,
                    embedding_b_regularizer,
                    margin,
                    epochs,
                    learning_rate)

    input_a = Input(shape=INPUT_SHAPE)
    input_b = Input(shape=INPUT_SHAPE)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)
    return model


def get_data():
    """
    Organizes the data into pairs for contrastive loss training
    :return:
    """
    # the data, shuffled and split between train and test sets
    X_train, subjects_train, _, _ = data_source.get_timeseries_data('train')
    X_test, subjects_test, _, _ = data_source.get_timeseries_data('test')

    y_train = subjects_train
    y_test = subjects_test

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # create training+test positive and negative pairs

    tr_pairs, tr_y = create_pairs(X_train, y_train)
    te_pairs, te_y = create_pairs(X_test, y_test)
    return tr_pairs, tr_y, te_pairs, te_y


def train(param_dict, save=True):

    def contrastive_loss(y, d):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        We want y and d to be different.
        Loss is 0 if y = 1 and d = 0
        Loss is 1 if y=d=1 or y=d=0
        '''
        margin = param_dict['margin']
        return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))

    model = create(param_dict)

    opt = RMSprop(lr=param_dict['learning_rate'])
    model.compile(loss=contrastive_loss, optimizer=opt, metrics=['accuracy'])

    tr_pairs, tr_y, te_pairs, te_y = get_data()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size=128,
              nb_epoch=param_dict['epochs'],
              callbacks=[early_stop])

    if save:
        # save as JSON
        json_string = model.to_json()
        with open(ARCHITECTURE_FILE, "w") as file:
            file.write(json_string)
        model.save_weights(WEIGHTS_FILE)

    return model



def get_embedding_function(trained_model):
    """
    Given a trained model, get the weights out that we need to construct only
    the embedding function
    :param trained_model:
    :return: an embedding function
    """
    # https://github.com/fchollet/keras/issues/41

    # Ugly but it gets the job done
    embedding = trained_model.layers[2]

    embedding_function = K.function([embedding.layers[0].input, K.learning_phase()], embedding.layers[-1].output)

    return embedding_function


def compute_auc_score(model, te_pairs, te_y):
    """
    This is the best measure of how the model is performing
    :param model:
    :return:
    """
    pred = 1 - model.predict([te_pairs[:, 0], te_pairs[:, 1]])[:, 0]
    return roc_auc_score(te_y, pred)




#
# subject_subset_max = np.unique(subjects_train)[4]
# idx = subjects_train.T[0] < subject_subset_max
# observations = X_train[idx]
# subjects = subjects_train[idx]


# Intermediate outputs seem to be broken in 1.0 :(


#
# embedding = embedding_function([observations])[0]
#
# print(embedding)
#
# x = embedding[:, 0]
# y = embedding[:, 1]
#
# print(x)
# print(y)
#
# plt.scatter(x, y, c=subjects)
# plt.savefig('foo.png', bbox_inches='tight')

if __name__ == '__main__':

    tr_pairs, tr_y, te_pairs, te_y = get_data()

    model = maybe_train(param_dict)
    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    f = get_embedding_function(model)

    X, subject, activity_one_hot, feature_names = data_source.get_timeseries_data()

    learning_phase = np.ones(X.shape[0])
    print (f([X, 1])[1:5])

    print(compute_auc_score(model, te_pairs, te_y))