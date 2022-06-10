import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
            _data = pd.read_csv(_data_dir + "/" + folder + "/" + file).values
            _data = scaler.fit_transform(_data)

            end_index = _step_size
            for i in range(0, len(_data) - end_index):
                _lstm_data.append(_data[i:i + end_index])
                _lstm_label.append(_label)

    # convert to numpy array
    _lstm_data = np.array(_lstm_data)
    _lstm_label = np.array(_lstm_label)
    _lstm_label = oe.fit_transform(_lstm_label.reshape(-1, 1)).toarray()

    return _lstm_data, _lstm_label


def set_impact_model(_step_size, _num_features, _num_labels):
    _model = Sequential()
    _model.add(Input((_step_size, _num_features)))  # input layer
    _model.add(LSTM(128))  # middle layer
    _model.add(Dense(_num_labels, activation="sigmoid"))  # output layer
    _model.compile(optimizer="adam", loss="binary_crossentropy")
    return _model


def create_model(_tr_data, _tr_label, _step_size, _num_features, _num_labels, _num_epochs, _batch_size, _model_name):
    _model = set_impact_model(_step_size, _num_features, _num_labels)
    _model.fit(_tr_data, _tr_label, batch_size=_batch_size, epochs=_num_epochs, shuffle=False)

    model_path = "../Models/" + _model_name
    _model.save(model_path)
    return _model


if __name__ == "__main__":
    names = ["im1.h5", "im2.h5", "im3.h5", "im4.h5"]
    cv = KFold(n_splits=4, random_state=1, shuffle=True)
    # dataset:
    # 133 video data -> 18 frame
    # step size = 12  -> 6 samples from each video
    # 9576 samples for training
    # 3192 samples for testing

    scores = {}
    data, labels = set_data_label("im_dat_csv", 12)

    for (train_index, test_index), name in zip(cv.split(data), names):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = create_model(X_train, y_train, 12, 99, 4, 100, 64, name)
        predicted = []

        print("predicting...")
        for test in X_test:
            pr = model.predict(np.array([test])).argmax(1)[0]
            predicted.append(pr)

        y_test = np.argmax(y_test, axis=1)

        print("scoring...")
        acc = accuracy_score(predicted, y_test)
        scores[name] = acc

    score = 0
    for name in names:
        score += scores[name]
        print(name, scores[name])
    print("\navg=", score/4)

# Results of current models:
#
# im1_12.h5 0.97
# im2_12.h5 0.98
# im3_12.h5 1.0
# im4_12.h5 0.9899497487437185
#
# avg= 0.9849874371859297
