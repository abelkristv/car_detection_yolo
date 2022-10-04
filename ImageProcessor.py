import cv2 as cv
import numpy as np

class ImageProcessor(object):
    def __init__(self):
        self._model = None
        self._predictions = None
        self._output = None
        self._class_ids = []
        self._confidences = []
        self._boxes = []

    @property
    def output(self):
        return self._output
    
    def predict(self, blob):
        self._model = cv.dnn.readNet('yolov5s.onnx')
        self._model.setInput(blob)
        self._predictions = self._model.forward()
        self._output = self._predictions

    def unwrap_detection(self, input_image, output_data):
        rows = output_data.shape[0]

        image_height, image_width ,_ = input_image.shape
        print(input_image.shape)

        x_factor = image_width / 640
        y_factor = image_height / 640

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:
                classes_scores = row[5:]
                _, _, _, max_indx = cv.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):
                    self._confidences.append(confidence)
                    self._class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])
                    self._boxes.append(box)

    def draw(self, image):
        indexes = cv.dnn.NMSBoxes(self._boxes, self._confidences, 0.25, 0.45)
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(self._confidences[i])
            result_class_ids.append(self._class_ids[i])
            result_boxes.append(self._boxes[i])

        class_list = []
        with open("classes.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        for i in range(len(result_class_ids)):
            box = result_boxes[i]
            class_id = result_class_ids[i]

            color = colors[class_id % len(colors)]

            conf = result_confidences[i]

            cv.rectangle(image, box, color, 2)
            cv.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv.putText(image, class_list[class_id], (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

            
