import pandas as pd
import os
import cv2
import mediapipe as mp

from additional_functions import set_paths, list_dir

# Path to train videos with annotation files
train_dir = "FallDataset/train"
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
    """
        This function uses given videos to create dataset using pose estimation.
            - results are stored in 'Datasets' folder.
            - cvs files named according to the RoomName and VideoName.

        @param : directly use global value 'train_dir'.
        @return: void
    """
    # prepare for pose estimation
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    points = mp_pose.PoseLandmark  # Landmarks

    # prepare dataset
    data_columns = []
    for p in points:
        x = str(p)[13:]
        data_columns.append(x + "_x")
        data_columns.append(x + "_y")
        data_columns.append(x + "_z")
    data_columns.append("label")

    # get room directories
    room_files = set_paths(train_dir, list_dir(train_dir))

    min_fall_time = 999  # just a big default value
    max_fall_time = 0  # just a small default value
    falls = []
    for room in room_files:
        # get video directory for room
        vid_dir = room + "/Videos"
        videos = set_paths(vid_dir, list_dir(vid_dir))

        # get annotation directory for room
        ann_dir = room + "/Annotation_files"

        # get fall start and fall end data from annotation file
        for vid in videos:
            # get corresponding annotation file
            ann = ann_dir + "/" + vid.split("/")[-1][:-3] + "txt"  # videos and annotations have the same same

            # read data from annotation file
            with open(ann) as _file:
                start = int(_file.readline())
                end = int(_file.readline())

                # set min_fall_time, max_fall_time
                fall_time = end - start
                falls.append(fall_time)
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
                            temp += [j.x, j.y, j.z]

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

            # set csv name
            folder_name = room.split("/")[-1]
            video_name = vid.split("/")[-1]
            csv_name = "Datasets/" + folder_name + "/" + video_name[:-3] + "csv"

            # create results directory if does not exists
            out_dir = "Datasets/" + folder_name
            if not os.path.exists(out_dir):
                if not os.path.exists("Datasets"):
                    os.mkdir("Datasets")
                os.mkdir(out_dir)

            # save the data as a csv file
            data.to_csv(csv_name, index=False)

    # save min, max and avg fall times
    avg_fall_time = sum(falls) / len(falls)
    fall_times = pd.DataFrame(columns=["min", "max", "avg"])
    fall_times.loc[0] = [min_fall_time, max_fall_time, avg_fall_time]
    fall_times.to_csv("Datasets/fall_times.csv", index=False)


if __name__ == "__main__":
    create_csv()
