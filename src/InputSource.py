# provide input for TheDetector class
import cv2


class InputSource:
    def __init__(self, input_source):
        self.__source = cv2.VideoCapture(input_source if input_source is not None else 0)  # todo: try if string 0 works
        self.__width = int(self.__source.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(self.__source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__length = int(self.__source.get(cv2.CAP_PROP_FRAME_COUNT))

        if not self.__source.isOpened():
            print("Unknown error opening ", ("camera" if input_source == "0" else "video"))
            exit(1)

    # return next frame from the input source
    def feed(self):
        while self.__source.isOpened():
            success, frame = self.__source.read()
            if success:
                yield frame
            else:
                self.__source.release()  # close video when it ends

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def length(self):
        return self.__length
