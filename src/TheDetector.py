# This is the final result where fall detection and impact prediction models will run together

import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from collections import Counter
from additional_functions import get_model
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

from InputSource import InputSource
from PoseEstimator import PoseEstimator

scaler = MinMaxScaler(feature_range=(0, 1))


class OutputCreator:
    def __init__(self):
        pass


class FallDetector:
    def __init__(self):
        pass


class ImpactPredictor:
    def __init__(self):
        pass


class TheDetector:
    def __init__(self, fall_model_name, impact_model_name):
        self.__fall_model = get_model(fall_model_name)
        # self.__impact_model = get_model(impact_model_name)
        self.__input_source: Optional[InputSource] = None
        self.__pose_estimator: Optional[PoseEstimator] = None

        self.__poses_for_fall_model = deque(maxlen=18)

    # this is the main function where everything runs
    # input_source: video name or nothing to use camera
    def run(self, input_source=None, save_result=False):
        self.initialize(input_source)

        for frame in self.__input_source.feed():
            if self.__pose_estimator.estimate(frame):  # True if there is a person in the view
                pose_est = self.__pose_estimator.get_pose_estimation
                frame = self.__pose_estimator.draw(frame)

            # set q as exit key
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            cv2.imshow('View', frame)
            cv2.setWindowProperty("View", cv2.WND_PROP_TOPMOST, 1)

    # make initial arrangements for detection
    def initialize(self, input_source):
        self.__input_source = InputSource(input_source)
        self.__pose_estimator = PoseEstimator(self.__input_source)


if __name__ == "__main__":
    the_detector = TheDetector(fall_model_name="v1_t1.h5", impact_model_name="")
    the_detector.run()
