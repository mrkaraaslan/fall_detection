from numpy import array
from collections import Counter
from additional_functions import get_model


class ImpactPredictor:
    def __init__(self, impact_model_name):
        self.__impact_model = get_model(impact_model_name)
        self.__final_decision = ""
        self.__switcher = {
            0: "Right",
            1: "Left",
            2: "Front",
            3: "Back"
        }

    def predict(self, pose_data):
        pose_data = pose_data[0]
        predictions = []
        for i in range(len(pose_data) - 12):
            dat = array([pose_data[i:i + 12]])
            predictions.append(self.__impact_model.predict(dat).argmax(1)[0])
        print("Impact predictions:", predictions)
        res = Counter(predictions).most_common(1)[0][0]
        self.__final_decision = self.__switcher.get(res)

    @property
    def prediction(self):
        return self.__final_decision

    def reset(self):
        self.__final_decision = ""
