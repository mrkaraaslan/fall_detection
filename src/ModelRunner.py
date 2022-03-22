import numpy as np
import os
from collections import deque

import cv2
import mediapipe as mp

from additional_functions import get_model

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


def run_model_video(model_name, step_size, video_path="0"):
    """
        Run model with the given video. Results will be saved under ../Model_run_results folder.

        @param video_path: absolute path to the test video.
        @param model_name: name of the model to test.
        @param step_size: step size of the model.
    """
    # load model
    model = get_model(model_name)

    # read video
    video_capture = cv2.VideoCapture(0 if video_path == "0" else video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video_capture.isOpened():
        print("Error opening the file")
        exit(1)

    # set output video
    out_name = video_path.split("/")[-1].split(".")[0]
    out_name += "_(" + model_name + ").mp4"
    out_video_path = "../Model_run_results/" + out_name

    if not os.path.exists("../Model_run_results"):
        os.mkdir("../Model_run_results")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 23.0, (width, height))

    # set mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils  # For drawing key-points
    points = mp_pose.PoseLandmark  # Landmarks

    counter = 1
    frames_for_model = deque(maxlen=step_size)
    while video_capture.isOpened():
        # just to see progress
        if counter % 100 == 0:
            print("\rProgress:{0}/{1}".format(counter, length), end='')

        frame_pose_est = []

        ret, frame = video_capture.read()
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(img_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for i, j in zip(points, landmarks):
                    frame_pose_est += [j.x, j.y, j.z]

                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            fall = "----"
            if len(frame_pose_est) != 0:
                frames_for_model.append(frame_pose_est)
            if len(frames_for_model) == 18:
                dat = np.asarray(frames_for_model)
                dat = scaler.fit_transform(dat)
                dat = np.array([dat])

                prediction = model.predict(dat).argmax(1)[0]
                if prediction == 1:
                    fall = "Detected Fall"
                else:
                    fall = "No fall"

            # specify the font and write using putText
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, fall, (10, height - 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # render as video
            out.write(frame)

            if video_path == "0":
                # set q as exit key
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                cv2.imshow('Camera', frame)
                cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)

            counter += 1
        else:
            break

    video_capture.release()
    out.release()
    if video_path == "0":
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # run_model_video("v1_t1.h5", 18, "test_source/50_Ways_to_Fall.mp4")
    # camera
    run_model_video("v1_t1.h5", 18)
