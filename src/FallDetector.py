from collections import deque, Counter
from additional_functions import get_model


class FallDetector:
    def __init__(self, fall_model_name):
        self.__fall_model = get_model(fall_model_name)
        self.__last_predictions = deque(maxlen=12)
        self.__final_decision = False
        self.__txt_decision = ""

    def detect(self, pose_data, body_angle):
        prediction = self.__fall_model.predict(pose_data).argmax(1)[0]
        self.__last_predictions.append(prediction)
        self.__txt_decision = "Detecting"
        ctr = Counter(self.__last_predictions)
        if ctr[1] > 9:  # there is a fall -> decide state: falling or fallen
            if body_angle >= 70:  # -> reject fall
                self.__txt_decision = "Daily"
                self.__final_decision = False
            elif body_angle >= 30:  # falling -> approve fall
                if self.__txt_decision != "Fallen":
                    self.__txt_decision = "Falling"
                self.__final_decision = True
            else:  # fallen -> approve fall
                self.__txt_decision = "Fallen"
                self.__final_decision = True

    @property
    def fall_detected(self):
        return self.__final_decision

    @property
    def state(self):
        return self.__txt_decision
