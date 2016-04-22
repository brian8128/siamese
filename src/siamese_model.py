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
from keras.layers.core import Dense, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Input

from settings import NB_EPOCH, EMBEDDING_DIM, LEARNING_RATE, OPTIMIZER, MARGIN, INPUT_SHAPE, PROJECT_HOME
from keras.models import model_from_json


from src import data_source
from src import activity_model

import os

ARCHITECTURE_FILE = "{}/saved_models/siamese_archetecture.json".format(PROJECT_HOME)
WEIGHTS_FILE = '{}/saved_models/siamese_model_weights.h5'.format(PROJECT_HOME)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def contrastive_loss(y, d):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We want y and d to be different.
    Loss is 0 if y = 1 and d = 0
    Loss is 1 if y=d=1 or y=d=0
    '''
    margin = MARGIN
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

def create_base_network_with_embedding():
    """
    Take the base network and add an embedding layer
    """

    # Borrow the base network from the activity prediction model
    base_network = activity_model.create_base_network(INPUT_SHAPE)

    embedding = Dense(EMBEDDING_DIM, activation='linear',
                      W_regularizer=l2(0.01),
                      b_regularizer=l2(0.01),
                      name='embedding'
                      )

    base_network.add(embedding)

    return base_network


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


def create():
    # network definition
    base_network = create_base_network_with_embedding()

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


def train():

    model = create()

    if OPTIMIZER is 'sgd':
        opt = SGD(lr=LEARNING_RATE)
    else:
        opt = RMSprop(lr=LEARNING_RATE)
    model.compile(loss=contrastive_loss, optimizer=opt, metrics=['accuracy'])

    tr_pairs, tr_y, te_pairs, te_y = get_data()

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size=128,
              nb_epoch=NB_EPOCH)

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
        model.compile(loss=contrastive_loss, optimizer=opt, metrics=['accuracy'])
        model.load_weights(WEIGHTS_FILE)

        print("Successfully loaded model from disk. No training needed.")
        return model

    except Exception:
        print("Unable to load model from disk. Training a new one")
        return train()

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

    # # compute final accuracy on training and test sets
    # pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    # tr_acc = compute_accuracy(pred, tr_y)
    # pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # te_acc = compute_accuracy(pred, te_y)
    #
    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    model = maybe_train()
    f = get_embedding_function(model)

    X, subject, activity_one_hot, feature_names = data_source.get_timeseries_data()

    learning_phase = np.ones(X.shape[0])
    print (f([X, 1]))