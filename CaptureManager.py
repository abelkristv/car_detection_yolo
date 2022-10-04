import cv2 as cv
import numpy as np
from ImageProcessor import ImageProcessor 

class CaptureManager(object):
    def __init__(self, cutHalf = False):
        self._frame = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None
        self._video = None
        self._fps = 0
        self._size = (0, 0)
        self._resized = None
        self._blob = None
        self._imageProcessor = None
        self._cutHalf = cutHalf
        self._tempFrame = None

    @property
    def blob(self):
        return self._blob

    def capture(self, filename):
        self._videoFileName = filename
        self._video = cv.VideoCapture(filename)
        self._fps = self._video.get(cv.CAP_PROP_FPS)
        self._size = (int(self._video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(self._video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
        success, frame = self._video.read()
        while success:
            self._frame = frame
            row, col, _ = self._frame.shape
            if self._cutHalf == True:
                self._tempFrame = np.copy(self._frame[:, int(col/2):])
                self._frame[:, int(col/2):] = (0, 0, 0)
            _max = max(col, row)
            self._resized = np.zeros((_max, _max, 3), np.uint8)
            self._resized[0:row, 0:col] = self._frame

            # resized to 640 * 640, normalize to [0,1] and swap red and blue channel
            self._blob = cv.dnn.blobFromImage(self._resized, 1/255.0, (640, 640), swapRB=True, crop=False)
            self._imageProcessor = ImageProcessor()
            self._imageProcessor.predict(self._blob)

            self._imageProcessor.unwrap_detection(self._resized, self._imageProcessor.output[0])

            self._imageProcessor.draw(self._frame)
            if self._cutHalf == True:
                print(self._tempFrame)
                self._frame[:, int(col/2):] = self._tempFrame

            cv.imshow('frame',self._frame)

            key = cv.waitKey(1)

            if key == 27:
                break

            success, frame = self._video.read()
            



