import numpy as np
import cv2
import os

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(
        im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE,
                FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(
        input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def post_process(input_image, outputs, classes):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Draw bounding box.
        cv2.rectangle(input_image, (left, top),
                      (left + width, top + height), BLUE, 3*THICKNESS)
        # Class label.
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        # Draw label.
        draw_label(input_image, label, left, top)
    return input_image


def main(imageFile, classesFile, modelWeights, showImage=False):
    # check if file exists.
    if not os.path.isfile(imageFile):
        print("File {} does not exist.".format(imageFile))
        exit()

    # Read the image.
    frame = cv2.imread(imageFile)

    # check if file exists.
    if not os.path.isfile(classesFile):
        print("File {} does not exist.".format(classesFile))
        exit()

    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # check if file exists.
    if not os.path.isfile(modelWeights):
        print("File {} does not exist.".format(modelWeights))
        exit()

    # Load the network.
    net = cv2.dnn.readNet(modelWeights)
    
    # Inference on the image.
    detections = pre_process(frame, net)
    
    # Run NMS and draw boxes.
    img = post_process(frame.copy(), detections, classes)
    
    """
    Put efficiency information. The function getPerfProfile returns the overall time for inference(t) 
    and the timings for each of the layers(in layersTimes).
    """
    t, _ = net.getPerfProfile()
    inferenceTime = 'Inference time: %.2f ms' % (
        t * 1000.0 / cv2.getTickFrequency())
    print(inferenceTime)

    if showImage:
        cv2.putText(img, inferenceTime, (20, 40), FONT_FACE, FONT_SCALE,
                    (0, 0, 255), THICKNESS, cv2.LINE_AA)

        # Show the image.
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)


if __name__ == '__main__':

    singleImage = False  # False for inference on multiple images
    showImage = True  # True to show image

    # Image file.
    imageFile = "images/zidane.jpg"

    # Image directory
    imageDir = "images"

    # Load class names
    classesFile = "config/coco.names"

    # Model weights file
    modelWeights = "weights/yolov5m.onnx"

    if singleImage:
        main(imageFile, classesFile, modelWeights, showImage)
        print("Done")

    else:
        # get all images in the directory
        imageFiles = []
        try:
            imageFiles = [os.path.join(imageDir, f) for f in os.listdir(
                imageDir) if os.path.isfile(os.path.join(imageDir, f))]
        except OSError as e:
            print(e)
            exit()

        for imageFile in imageFiles:
            main(imageFile, classesFile, modelWeights, showImage)

        print("Done")
