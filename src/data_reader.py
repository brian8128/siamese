from settings import PROJECT_HOME
import numpy as np
import pandas as pd


def get_data(train=True):
    if train:
        data_set = 'train'
    else:
        data_set = 'test'

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

    # # The Inertial Signals folder contains only timeseries data.  This might be easier
    # # to apply convolutional layers to.
    # df = pd.read_csv(data_dir + "Inertial Signals/body_gyro_x_train.txt", sep='\s+')
    # body_gyro = df.values

    idx = activity.T[0] < 4

    y_subject = y_subject[idx]
    X = X[idx]

    perm = np.random.permutation((range(X.shape[0])))

    return X[perm], y_subject[perm]
