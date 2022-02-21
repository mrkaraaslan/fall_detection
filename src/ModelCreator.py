import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from contextlib import suppress
from collections import Counter

import cv2
import mediapipe as mp

import tensorflow as tf

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from additional_functions import list_dir, set_paths, eliminate_files

scaler = MinMaxScaler(feature_range=(0, 1))
oe = OneHotEncoder()

# global arrangements
# todo: remove global variables
num_features = 99
num_outputs = 2


def get_label(label_arr):
    """
        @param label_arr: label array of step size long.
        @return
            is_valid: is given label array valid or not:
                - valid if 2/3 is 1
                - not valid otherwise
            _label: dominant label
    """
    _label = None
    is_valid = True
    counts = Counter(label_arr)

    if 1 in counts and 0 in counts:
        if counts[1] >= counts[0] * 2:
            _label = 1
        else:
            is_valid = False
    elif 1 in counts:
        _label = 1
    else:
        _label = 0

    return is_valid, _label


def set_data_label(main_dir, step_size):
    """
        Creates dataset and corresponding labels to use in lstm model with given step_size.

        @param main_dir: main directory of dataset directories.
        @param step_size: step size of lstm model.
        @return
            lstm_data: data to be used in lstm model.
            lstm_label: corresponding labels for data.
    """
    # set directory paths
    dirs = eliminate_files(set_paths(main_dir, list_dir(main_dir)))

    # set file paths
    all_files = []
    for _dir in dirs:
        all_files += set_paths(_dir, list_dir(_dir))

    # set dataset for LSTM
    lstm_data = []
    lstm_label = []

    for _file in all_files:
        start_index = 0

        # separate data and label
        _file_data = pd.read_csv(_file)
        _file_label = _file_data["label"].values
        _file_data = _file_data.drop(["label"], axis=1).values

        # scale data
        _file_data = scaler.fit_transform(_file_data)

        while start_index + step_size < len(_file_data):
            sub_data = _file_data[start_index: start_index + step_size]
            sub_label = _file_label[start_index: start_index + step_size]
            start_index += 1

            # decide label for label array
            is_valid, f_label = get_label(sub_label)
            if is_valid:
                lstm_data.append(sub_data)
                lstm_label.append(sub_label)

    # convert to numpy array
    lstm_data = np.array(lstm_data)
    lstm_label = np.array(lstm_label)

    # use one hot encoding on label array
    lstm_label = oe.fit_transform(lstm_label.reshape(-1, 1)).toarray()

    return lstm_data, lstm_label


if __name__ == "__main__":
    lstm_step_size = 18
