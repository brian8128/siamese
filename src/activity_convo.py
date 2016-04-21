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

from src.models import train_activity_model
from src.data_source import get_timeseries_data

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


model = train_activity_model()
X_test, subject_test, activity_test, _ = get_timeseries_data('test')

pred = model.predict(X_test)

score = model.evaluate(X_test, activity_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Looks like subject 10 is the subject we get wrong most often, make
# a confusion matrix just for him/her



predicted_classes = np.expand_dims(np.argmax(pred, axis=1) + 1, axis=1)
activity_test_classes = (np.argmax(activity_test, axis=1) + 1).getA()

# incorrect_prediction = (predicted_classes != activity_test)
subject_10 = subject_test == 10

cm = confusion_matrix(activity_test_classes, predicted_classes)
cm_10 = confusion_matrix(activity_test_classes[subject_10], predicted_classes[subject_10])


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


print(model.layers[0].get_weights()[0].shape)