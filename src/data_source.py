from __future__ import division, print_function, absolute_import
from settings import PROJECT_HOME
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_data(data_set='train'):

    data_dir = "{0}/data/UCI_HAR_Dataset/{1}/".format(PROJECT_HOME, data_set)
    X_file = 'X_{}.txt'.format(data_set)
    activity_file = 'y_{}.txt'.format(data_set)
    subject_file = 'subject_{}.txt'.format(data_set)

    df = pd.read_csv(data_dir + X_file, header=None, sep='\s+')
    X = df.values

    df = pd.read_csv(data_dir + activity_file, header=None)
    activity = df.values

    df = pd.read_csv(data_dir + subject_file, header=None)
    y_subject = df.values

    # Restrict to walking activities.  Walking upstairs, walking downstairs and just walking
    idx = activity.T[0] < 4

    y_subject = y_subject[idx]
    X = X[idx]

    # Shuffle the data before returning
    perm = np.random.permutation((range(X.shape[0])))
    return X[perm], y_subject[perm]


def get_timeseries_data(data_set='train', one_hot=True):
    """
    :param data_set: 'train' for train data or 'test' for test data
    :return: tuple containing the gyro data, the subject id, and the activity id
             data are shuffled before returning.
    """

    data_dir = "{0}/data/UCI_HAR_Dataset/{1}/".format(PROJECT_HOME, data_set)
    timeseries_features = []
    timeseries_feature_names = []
    for feature in ['body_acc',
                    'body_gyro',
                    #'total_acc'
                    ]:
        for dim in ['x', 'y', 'z']:
            data_file = 'Inertial Signals/{0}_{1}_{2}.txt'.format(feature, dim, data_set)
            df = pd.read_csv(data_dir + data_file, header=None, sep='\s+')
            # We will want 128 x 1 data with 9 channels, not 128 x 9 data.
            series = np.expand_dims(np.expand_dims(df.values, axis=1), axis=3)
            timeseries_features.append(series)

            timeseries_feature_names.append("_".join([feature, dim]))

    X = np.concatenate(timeseries_features, axis=1)

    activity_file = 'y_{}.txt'.format(data_set)
    subject_file = 'subject_{}.txt'.format(data_set)

    df = pd.read_csv(data_dir + activity_file, header=None)
    activity = df.values

    df = pd.read_csv(data_dir + subject_file, header=None)
    subject = df.values

    # Restrict to walking activities.  Walking upstairs, walking downstairs and just walking
    idx = activity.T[0] < 4
    subject = subject[idx]
    activity = activity[idx]
    X = X[idx]

    # Scale the data:
    for i in range(X.shape[1]):
        series = X[:, i, :, 0]
        mean = np.mean(series)
        scaling_factor = np.percentile(np.abs(series-mean), 90)
        X[:, i, :, 0] = (series - mean) / scaling_factor

        # X.shape
        # (3285, 9, 128, 1)

    # Shuffle the data before returning
    perm = np.random.permutation((range(X.shape[0])))

    encoder = OneHotEncoder()
    # Will using the fit transform for both test and train be a problem?
    # It will certainly become very evident if it is
    activity_one_hot = encoder.fit_transform(activity[perm]).todense()

    return X[perm], subject[perm], activity_one_hot, map(lambda x: x[5:], timeseries_feature_names)


if __name__ == '__main__':
    X, subjects, activities, feature_names = get_timeseries_data('test')

    encoder = OneHotEncoder()
    activity_train_one_hot = encoder.fit_transform(activities).todense()

    print(np.sum(activity_train_one_hot, axis=0))