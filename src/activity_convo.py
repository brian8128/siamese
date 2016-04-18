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
from settings import NB_EPOCH, NB_CONV_FILTERS, DROPOUT
from src import data_reader
from src.base_network import create_base_network
from sklearn.metrics import confusion_matrix

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
nb_epoch = NB_EPOCH

# network definition
model = create_base_network(input_shape)
# output layer
model.add(Dense(3, activation='softmax'))

# train
opt = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train,
      validation_data=(X_test, y_test),
      batch_size=128,
      nb_epoch=nb_epoch)

pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted_classes = np.expand_dims(np.argmax(pred, axis=1) + 1, axis=1)
cm = confusion_matrix(activity_test, predicted_classes)
print(cm)
