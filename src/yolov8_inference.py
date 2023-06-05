import argparse

import cv2.dnn
import numpy as np

from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_kevin_ravi/robocup_2023_dataset_308_461/dataset.yaml'))['names']
# CLASSES = yaml_load(check_yaml('/home/kvnptl/work/yolo-inference-onnx/config/coco.yaml'))['names']


colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='yolov8n.onnx', help='Input your onnx model.')
    # parser.add_argument('--img', default=str(ROOT / 'assets/bus.jpg'), help='Path to input image.')
    # args = parser.parse_args()
    modelWeights = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_kevin_ravi/models/best.onnx"
    # modelWeights = "/home/kvnptl/work/yolo-inference-onnx/weights/yolov5m.onnx"

    imageFile = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_kevin_ravi/combined_308_461/YOLODataset/train/images/set1_frame00308.png"
    # imageFile = "/home/kvnptl/work/yolo-inference-onnx/images/zidane.jpg"

    # main(args.model, args.img)
    main(modelWeights, imageFile)