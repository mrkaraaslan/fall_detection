# This is the final result where fall detection and impact prediction models will run together

import cv2
from numpy import asarray, array
# import mediapipe  # uncomment this line if you want to use mediapipe
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

from InputSource import InputSource
from PoseEstimator import PoseEstimator
from FallDetector import FallDetector
from ImpactPredictor import ImpactPredictor

scaler = MinMaxScaler(feature_range=(0, 1))


class OutputCreator:
    def __init__(self):
        pass


class TheDetector:
    def __init__(self, fall_model_name, impact_model_name):
        self.__input_source: Optional[InputSource] = None
        self.__pose_estimator: Optional[PoseEstimator] = None
        self.__fall_detector = FallDetector(fall_model_name)
        self.__impact_predictor = ImpactPredictor(impact_model_name)

        self.__poses_for_fall_model = deque(maxlen=18)
        self.__none_counter = 0

    # this is the main function where everything runs
    # input_source: video name or nothing to use camera
    def run(self, input_source=None):  # , save_result=False):
        self.initialize(input_source)

        one_for_each_fall = True  # not to call impact prediction model multiple times
        for frame in self.__input_source.feed():
            if self.__pose_estimator.estimate(frame):  # True if there is a person in the view
                self.__none_counter = 0
                self.__poses_for_fall_model.append(self.__pose_estimator.pose_estimation)
                frame = self.__pose_estimator.draw(frame)
            else:
                self.__none_counter += 1
                if self.__none_counter == 9:
                    self.__none_counter = 0
                    self.__poses_for_fall_model.clear()
                    self.__fall_detector.reset()

            if len(self.__poses_for_fall_model) == 18:
                # send to fall detection model
                data = asarray(self.__poses_for_fall_model)
                data = scaler.fit_transform(data)
                data = array([data])
                self.__fall_detector.detect(data, self.__pose_estimator.body_angle)

                if self.__fall_detector.fall_detected:
                    # send poses_for_model to impact prediction model by looking
                    # the state of fall detection model
                    # it can be early to send when the state is "Falling"
                    if self.__fall_detector.state == "Fallen":
                        if one_for_each_fall:
                            # send to impact prediction model
                            self.__impact_predictor.predict(data)
                            print(self.__impact_predictor.prediction)
                            one_for_each_fall = False
                else:
                    one_for_each_fall = True
                    self.__impact_predictor.reset()

            # write states on the frame
            txt = self.__fall_detector.state + " " + self.__impact_predictor.prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, txt, (10, self.__input_source.height - 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # set q as exit key
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # show the frame
            cv2.imshow('View', frame)
            cv2.setWindowProperty("View", cv2.WND_PROP_TOPMOST, 1)

    # make initial arrangements for detection
    def initialize(self, input_source):
        self.__input_source = InputSource(input_source)
        self.__pose_estimator = PoseEstimator(self.__input_source)

        # reset is needed when you run for multiple videos one after another
        self.__fall_detector.reset()
        self.__impact_predictor.reset()
        self.__poses_for_fall_model.clear()
        self.__none_counter = 0


if __name__ == "__main__":
    the_detector = TheDetector(fall_model_name="v1_t1.h5", impact_model_name="im3_12.h5")
    the_detector.run("test_source/50_Ways_to_Fall.mp4")
