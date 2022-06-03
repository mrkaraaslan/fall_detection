import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from additional_functions import list_dir

scaler = MinMaxScaler(feature_range=(0, 1))
oe = OneHotEncoder()


def set_data_label(_data_dir, _step_size):
    folders = list_dir(_data_dir)
    _lstm_data = []
    _lstm_label = []
    for folder in folders:
        _label = folder[0]
        files = list_dir(_data_dir + "/" + folder)
        for file in files:
            data = pd.read_csv(_data_dir + "/" + folder + "/" + file).values
            data = scaler.fit_transform(data)

            end_index = _step_size
            for i in range(0, len(data) - end_index):
                _lstm_data.append(data[i:i + end_index])
                _lstm_label.append(_label)

    # convert to numpy array
    _lstm_data = np.array(_lstm_data)
    _lstm_label = np.array(_lstm_label)
    _lstm_label = oe.fit_transform(_lstm_label.reshape(-1, 1)).toarray()

    return _lstm_data, _lstm_label


def set_impact_model(_step_size, _num_features, _num_labels):
    model = Sequential()
    model.add(Input((_step_size, _num_features)))  # input layer
    model.add(LSTM(128))  # middle layer
    model.add(Dense(_num_labels, activation="sigmoid"))  # output layer
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def create_impact_model(_dataset_dir, _step_size, _num_features, _num_labels, _num_epochs, _batch_size, _model_name):
    # set data
    lstm_data, lstm_label = set_data_label(_dataset_dir, _step_size)

    # set model
    lstm_model = set_impact_model(_step_size, _num_features, _num_labels)

    # train model
    lstm_model.fit(lstm_data, lstm_label, batch_size=_batch_size, epochs=_num_epochs, shuffle=False)

    # save model
    model_path = "../Models/" + _model_name
    lstm_model.save(model_path)


if __name__ == "__main__":
    # what do I need
    # - dataset directory -> im_dat/train/
    # - step size -> try 12
    # - number of features -> 99
    # - number of labels -> 4
    # - number of epochs -> try 100
    # - batch size -> try 64
    # - model name -> impact_model.h5
    # - model directory -> ../Models/
    create_impact_model("im_dat/train", 12, 99, 4, 100, 64, "impact_model.h5")
