"""
Trains a convolutional neural network to predict whether someone is walking upstairs,
downstairs or walking not on stairs using normal categorical cross entropy.
Attains 95% accuracy after 3 minutes of local training.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1)  # for reproducibility

from keras.layers.core import Dense
from keras.optimizers import SGD, RMSprop, Adam

from sklearn.preprocessing import OneHotEncoder
from settings import NB_EPOCH, LEARNING_RATE
from src import data_reader
from src.base_network import create_base_network
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Scaled and shuffled data
X_train, subject_train, activity_train, _ = data_reader.get_timeseries_data('train')
X_test, subject_test, activity_test, _ = data_reader.get_timeseries_data('test')

encoder = OneHotEncoder()
activity_train_one_hot = encoder.fit_transform(activity_train).todense()
activity_test_one_hot = encoder.transform(activity_test).todense()

y_train = activity_train_one_hot
y_test = activity_test_one_hot

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

input_shape = (X_train.shape[1], 128, 1)
nb_epoch = NB_EPOCH

# network definition
model = create_base_network(input_shape)
# output layer
model.add(Dense(3, activation='softmax'))

# train
opt = RMSprop(lr=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train,
      validation_data=(X_test, y_test),
      batch_size=128,
      nb_epoch=nb_epoch)

pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Looks like subject 10 is the subject we get wrong most often, make
# a confusion matrix just for him/her
predicted_classes = np.expand_dims(np.argmax(pred, axis=1) + 1, axis=1)
incorrect_prediction = (predicted_classes != activity_test)
subject_10 = subject_test == 10

predicted_classes_10 = predicted_classes[subject_10]
activity_test_10 = activity_test[subject_10]

cm = confusion_matrix(activity_test, predicted_classes)
cm_10 = confusion_matrix(activity_test_10, predicted_classes_10)


def plot_confusion_matrix(cm, filename, title, cmap=plt.cm.YlGnBu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Walking', 'Walking Upstairs', 'Walking Downstairs'], rotation=45)
    plt.yticks(tick_marks, ['Walking', 'Walking Upstairs', 'Walking Downstairs'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('images/{}'.format(filename), bbox_inches='tight')
    plt.clf()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_10_normalized = cm_10.astype('float') / cm_10.sum(axis=1)[:, np.newaxis]


print(cm)

plot_confusion_matrix(cm_normalized, 'activity_prediction_confusion_matrix.png', "All Test Subjects")
plot_confusion_matrix(cm_10_normalized, 'activity_prediction_confusion_matrix_10.png', "Subject 10 Only")


