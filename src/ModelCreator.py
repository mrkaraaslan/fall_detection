import numpy as np
import pandas as pd
import os
from collections import Counter

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from additional_functions import list_dir, set_paths, eliminate_files

scaler = MinMaxScaler(feature_range=(0, 1))
oe = OneHotEncoder()


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


def set_model(step_size, num_features, num_outputs):
    model = Sequential()
    model.add(Input((step_size, num_features)))  # input layer
    model.add(LSTM(128))  # middle layer
    model.add(Dense(num_outputs, activation="sigmoid"))  # output layer
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def train_model(dataset_dir, step_size, num_features, num_out, num_epochs, batch_size, model_name):
    # set data
    lstm_data, lstm_label = set_data_label(dataset_dir, step_size)

    # set model
    lstm_model = set_model(step_size, num_features, num_out)

    # train model
    lstm_model.fit(lstm_data, lstm_label, batch_size=batch_size, epochs=num_epochs, shuffle=False)

    # save model
    model_path = "Models/" + model_name
    if not os.path.exists("Models"):
        os.mkdir("Models")
    lstm_model.save(model_path)


if __name__ == "__main__":
    epochs = 100
    batch = 64
    name = "v1_t1.h5"

    train_model("FallDataset/train", 18, 99, 2, epochs, batch, name)
