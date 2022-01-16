import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from contextlib import suppress
import cv2
import mediapipe as mp
import time

from additional_functions import set_paths, list_dir

# Path to train videos with annotation files
train_dir = "FallDetection/train"
# file system
# train_dir
# --RoomName1
# ----Videos
# ------various videos
# ----Annotation_files
# ------annotation of videos
# --RoomName2
# ----same as 1
# and more ...


def create_csv():
    # prepare for pose estimation
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
    points = mpPose.PoseLandmark  # Landmarks

    # prepare dataset
    data_columns = []
    for p in points:
        x = str(p)[13:]
        data_columns.append(x + "_x")
        data_columns.append(x + "_y")
        data_columns.append(x + "_z")
        data_columns.append(x + "_vis")  # visibility score
    data_columns.append("label")

    # get room directories
    room_files = set_paths(train_dir, list_dir(train_dir))

    min_fall_time = 999  # just a big default value
    max_fall_time = 0  # just a small default value
    for room in room_files:
        # get video directory for room
        vid_dir = room + "/Videos"
        videos = set_paths(vid_dir, list_dir(vid_dir))

        # get annotation directory for room
        ann_dir = room + "/Annotations"
        annotations = set_paths(ann_dir, list_dir(ann_dir))

        # get fall start and fall end data from annotation file
        for vid, ann in zip(videos, annotations):
            # read data from annotation file
            with open(ann) as _file:
                start = int(_file.readline())
                end = int(_file.readline())

                # set min_fall_time, max_fall_time
                fall_time = end - start
                if fall_time != 0:
                    if fall_time < min_fall_time:
                        min_fall_time = fall_time
                    if fall_time > max_fall_time:
                        max_fall_time = fall_time

            # pose estimation
            data = pd.DataFrame(columns=data_columns)  # Empty dataset
            video = cv2.VideoCapture(vid)  # capture video

            frame_counter = 1
            row_counter = 0
            while video.isOpened():
                success, frame = video.read()  # get video frame by frame

                if success:
                    temp = []
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert video to RGB

                    results = pose.process(img_rgb)  # make pose estimation

                    # fill dataset
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        for i, j in zip(points, landmarks):
                            temp += [j.x, j.y, j.z, j.visibility]

                        label = 0
                        if start < frame_counter < end:
                            label = 1  # means fall

                        temp += [label]

                        data.loc[row_counter] = temp
                        row_counter += 1
                    frame_counter += 1
                else:  # close video
                    break

            video.release()

            # create results directory if does not exists
            out_dir = room + "/results"
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            csv_name = vid.replace("Videos", "results")[:-4] + ".csv"
            data.to_csv(csv_name, index=False)  # save the data as a csv file

    # save min and max fall times
    fall_times = pd.DataFrame(columns=["min_fall_time", "max_fall_time"])
    fall_times.loc[0] = [min_fall_time, max_fall_time]
    if not os.path.exists("Datasets"):
        os.mkdir("Datasets")
    fall_times.to_csv("Datasets/fall_times.csv", index=False)


def collect_csv():
    pass


if __name__ == "__main__":
    create_csv()
    collect_csv()
