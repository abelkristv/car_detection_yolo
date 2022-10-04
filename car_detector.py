import cv2 as cv
import numpy as np
import argparse
from CaptureManager import CaptureManager
from ImageProcessor import ImageProcessor

class CarDetectorApp(object):
    def __init__(self):
        self._capture = None
        self._imageProcessor = None
        self._blob = None
        self._cutHalf = False

    def run(self):
        self.parser()
        self._capture = CaptureManager(self._cutHalf)
        self._capture.capture('bridge.mp4')
        
    def parser(self):
        parserArgs = argparse.ArgumentParser()
        parserArgs.add_argument('--cutHalf', action="store_true")
        args = parserArgs.parse_args()
        if args.cutHalf:
            self._cutHalf = True

if __name__=="__main__":
    CarDetectorApp().run()
        
        
