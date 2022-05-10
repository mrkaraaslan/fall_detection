import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from collections import Counter
from additional_functions import get_model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


class FallDetector:
    def __init__(self, model_name):
        self.__model_name = model_name
        self.__model = get_model(model_name)

    def detect(self, video_path=None, with_output=False, show_camera=False):
        if video_path is None:
            video_path = "0"

        # open video or camera if video path is not given
        cap = cv2.VideoCapture(video_path if video_path != "0" else 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            print("Error opening", ("camera" if video_path == "0" else "video"))
            exit(1)

        out = None
        if with_output:
            # set output video
            out_name = video_path.split("/")[-1].split(".")[0]
            out_name += "_(" + self.__model_name + "_detector).mp4"
            out_video_path = "../Model_run_results/" + out_name

            i = 1
            while os.path.exists(out_video_path):
                out_name = video_path.split("/")[-1].split(".")[0]
                out_name += "_(" + self.__model_name + "_detector)" + str(i) + ".mp4"
                out_video_path = "../Model_run_results/" + out_name
                i += 1

            if not os.path.exists("../Model_run_results"):
                os.mkdir("../Model_run_results")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_path, fourcc, 23.0, (width, height))

        # set mediapipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_draw = mp.solutions.drawing_utils  # For drawing key points

        frames_for_model = deque(maxlen=18)
        last_predictions = deque(maxlen=18)
        state = ""

        none_counter = 0
        frame_counter = 0
        while cap.isOpened():
            # to follow progress
            frame_counter += 1
            if frame_counter % 20 == 0:
                print("\rProgress:{0}|{1}".format(frame_counter, (length if video_path != "0" else "∞")), end='')

            # read one frame from video or camera
            success, frame = cap.read()
            if success:  # view is successfully taken
                # make pose estimation
                pose_est = []
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                if results.pose_landmarks:  # there is a person in the view
                    none_counter = 0

                    landmarks = results.pose_landmarks.landmark

                    # set points data
                    for i in landmarks:
                        pose_est += [i.x, i.y, i.z]
                    frames_for_model.append(pose_est)

                    # to detect shape
                    knee_x = int((pose_est[75] + pose_est[78]) / 2 * width)
                    knee_y = int((pose_est[76] + pose_est[79]) / 2 * height)
                    bottom_point = (knee_x, knee_y)  # middle of knees l: (75,76)  r:(78,79)
                    top_point = (int(pose_est[0] * width), int(pose_est[1] * height))  # nose
                    rec_h = abs(top_point[1] - bottom_point[1])
                    rec_w = abs(top_point[0] - bottom_point[0])
                    rec_w += 1 if rec_w == 0 else 0
                    body_angle = np.rad2deg(np.arctan(rec_h / rec_w))

                    # draw pose estimation and shape rectangle
                    if with_output or (show_camera and video_path == "0"):
                        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        cv2.rectangle(frame, bottom_point, top_point, (0, 255, 0), 2)

                    if len(frames_for_model) == frames_for_model.maxlen:
                        # prepare data for model
                        dat = np.asarray(frames_for_model)
                        dat = scaler.fit_transform(dat)
                        dat = np.array([dat])

                        prediction = self.__model.predict(dat).argmax(1)[0]
                        # using predictions and body shape final decision will be made.
                        last_predictions.append(prediction)
                        prediction_counts = Counter(last_predictions)

                        the_decision = False  # final decision -> keep this till the end even if it is unnecessary
                        if prediction_counts[1] >= last_predictions.maxlen / 3 * 2:
                            the_decision = True  # prediction -> fallen
                        else:  # prediction -> not fallen
                            if not (state == "Fallen" and body_angle <= 70):
                                state = "Daily"

                        if the_decision:
                            if body_angle >= 70:  # standing -> reject fall
                                the_decision = False
                                state = "Daily"
                            elif body_angle >= 30:  # falling -> approve fall
                                if state != "Fallen":
                                    state = "Falling"
                            else:  # fallen -> approve fall
                                state = "Fallen"

                        # put state on the result
                        if with_output or (video_path == 0 and show_camera):
                            # specify the font and write using putText
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            txt = str(frame_counter) + " " + state + " " + str(the_decision) + " " + str(body_angle)
                            cv2.putText(frame, txt, (10, height - 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:  # no one in the view
                    # if view is empty clear old data
                    none_counter += 1
                    if none_counter == frames_for_model.maxlen / 3 * 2:
                        none_counter = 0
                        frames_for_model.clear()
                        last_predictions.clear()
                        state = ""

                if with_output:
                    # render as video
                    out.write(frame)

                if video_path == "0" and show_camera:
                    # set q as exit key
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                    cv2.imshow('Camera', frame)
                    cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
            else:  # video finished
                break


if __name__ == "__main__":
    detector = FallDetector("v1_t1.h5")
    detector.detect(video_path="test_source/50_Ways_to_Fall.mp4", with_output=True)
