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

scaler = MinMaxScaler(feature_range=(0, 1))


class PoseEstimator:
    def __init__(self, input_source):
        self.__mp_pose = mp.solutions.pose
        self.__pose = self.__mp_pose.Pose()  # to make pose estimation
        self.__mp_draw = mp.solutions.drawing_utils  # For drawing key points

        self.__pose_landmarks = None  # save for later use in draw operation
        self.__pose_est = []
        self.__body_angle = 0
        self.__box_bottom_point = (0, 0)  # to draw box
        self.__box_top_point = (0, 0)  # to draw box
        self.__input_data_provider = input_source  # to get frame data

    def estimate(self, frame):
        # reset necessary variables
        self.__pose_landmarks = None
        self.__pose_est = []

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB

        _sub_result = self.__pose.process(img_rgb)  # make pose estimation
        if _sub_result.pose_landmarks:  # there is a person in the view
            self.__pose_landmarks = _sub_result.pose_landmarks  # save for later use in draw operation

            # set points data
            landmarks = _sub_result.pose_landmarks.landmark
            for i in landmarks:
                self.__pose_est += [i.x, i.y, i.z]
                # todo: here or later I should collect these estimations for fall detection model

            # to detect body angle and draw box if necessary
            knee_x_middle = int((self.__pose_est[75] + self.__pose_est[78]) / 2 * self.__input_data_provider.width)
            knee_y_middle = int((self.__pose_est[76] + self.__pose_est[79]) / 2 * self.__input_data_provider.height)
            self.__box_bottom_point = (knee_x_middle, knee_y_middle)  # middle of knees l: (75,76)  r:(78,79)
            nose_x = int(self.__pose_est[0] * self.__input_data_provider.width)
            nose_y = int(self.__pose_est[1] * self.__input_data_provider.height)
            self.__box_top_point = (nose_x, nose_y)  # nose
            rec_h = abs(self.__box_top_point[1] - self.__box_bottom_point[1])
            rec_w = abs(self.__box_top_point[0] - self.__box_bottom_point[0])
            rec_w += 1 if rec_w == 0 else 0
            self.__body_angle = np.rad2deg(np.arctan(rec_h / rec_w))

        else:  # no one in the view
            pass
            # todo: decide what to do, return false maybe

    def draw(self, frame):
        if self.__pose_landmarks is not None:
            self.__mp_draw.draw_landmarks(frame, self.__pose_landmarks, self.__mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(frame, self.__box_bottom_point, self.__box_top_point, (0, 255, 0), 2)
        return frame

    @property
    def get_pose_estimation(self):
        return self.__pose_est


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

    # this is the main function where everything runs
    # input_source: video name or nothing to use camera
    def run(self, input_source=None, save_result=False):
        self.initialize(input_source)

        for frame in self.__input_source.feed():
            self.__pose_estimator.estimate(frame)
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
