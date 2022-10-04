import cv2 as cv
import numpy as np

model = cv.dnn.readNet('yolov5s.onnx')

def format_yolov5(source):
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source

    # resized to 640* 640, normalize to [0,1] and swap red and blue channel
    result = cv.dnn.blobFromImage(resized, 1/255.0, (640,640), swapRB=True)
    return result

image = cv.imread('bridge.jpg')
blob = format_yolov5(image)
model.setInput(blob)
predictions = model.forward()
output = predictions[0]
#print(output[0])

def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        #print(confidence)
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                
                confidences.append(confidence)

                class_ids.append(class_id)
                
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes

class_ids, confidences, boxes = unwrap_detection(image, output)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

result_class_ids = []
result_confidences = []
result_boxes = []

for i in indexes:
    result_confidences.append(confidences[i])
    result_class_ids.append(class_ids[i])
    result_boxes.append(boxes[i])

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
    cv.putText(image, class_list[class_id], (box[0] + 5, box[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

cv.imshow('Image', image)
cv.waitKey()
