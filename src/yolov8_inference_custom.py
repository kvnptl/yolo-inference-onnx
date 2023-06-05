import numpy as np
import cv2
import os
from tqdm import tqdm

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)


def draw_label(im, label, x, y, color):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle on top of the rectangle.
    cv2.rectangle(
        im, (x, y-20), (x + dim[0], y), BLACK, cv2.FILLED)
    # Display text inside the BLACK rectangle.
    cv2.putText(im, label, (x, y - 5), FONT_FACE,
                FONT_SCALE, color, THICKNESS)


def pre_process(input_image, net):

    [height, width, _] = input_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = input_image

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1/255,  size=(INPUT_WIDTH, INPUT_HEIGHT), swapRB=True)

    # Sets the input to the network.
    net.setInput(blob)

    # Run the forward pass to get output of the output layers.
    outputs = net.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    return outputs


def post_process(input_image, outputs, classes):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs.shape[1]
    
    [height, width, _] = input_image.shape
    length = max((height, width))
    scale = length / 640

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Iterate through detections.
    for r in range(rows):
        classes_scores = outputs[0][r][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

        # Discard bad detections and continue.
        if maxScore >= CONFIDENCE_THRESHOLD:
            x =  outputs[0][r][0] - (0.5 * outputs[0][r][2]) # x is left of the box
            y =  outputs[0][r][1] - (0.5 * outputs[0][r][3]) # y is the top of the box
            w = outputs[0][r][2] # width of the box
            h = outputs[0][r][3] # height of the box
            box = [x, y, w, h]
            boxes.append(box)
            confidences.append(maxScore)
            class_ids.append(maxClassIndex)
            

    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, SCORE_THRESHOLD)
    for i in range(len(indices)):
        index = indices[i]
        box = boxes[index]
        # Scale bounding box coordinates based on resizing factor.
        x = round(box[0] * scale)
        y = round(box[1] * scale)
        w = round((box[0] + box[2]) * scale)
        h = round((box[1] + box[3]) * scale)

        color = colors[class_ids[index]]

        # Draw bounding box.
        cv2.rectangle(input_image, (x, y), (w, h), color, 2)
        # Class label.
        label = "{}:{:.2f}".format(classes[class_ids[index]], confidences[i])
        # Draw label.
        draw_label(input_image, label, x, y, color)

    return input_image


def main(imageFile, classesFile, modelWeights):
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
    net = cv2.dnn.readNetFromONNX(modelWeights)
    
    # Inference on the image.
    detections = pre_process(frame, net)
    
    # Run NMS and draw boxes.
    img = post_process(frame.copy(), detections, classes)
    
    """
    Put efficiency information. The function getPerfProfile returns the overall time for inference(t) 
    and the timings for each of the layers(in layersTimes).
    """
    inferenceTime, _ = net.getPerfProfile()
    inferenceTime = inferenceTime * 1000.0 / cv2.getTickFrequency()
    
    return img, inferenceTime


if __name__ == '__main__':

    singleImage = False  # False for inference on multiple images
    showImage = False  # True to show image
    createVideo = True  # True to create a video on all predicted images (only if singleImage is False)

    # Image file.
    imageFile = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_kevin_ravi/combined_308_461/YOLODataset/train/images/set1_frame00308.png"

    # Image directory
    imageDir = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_gokul_vivek/test_images"

    # Load class names
    classesFile = "/home/kvnptl/work/yolo-inference-onnx/config/robocup_2023.names"

    # Model weights file
    modelWeights = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/dataset_collection_kevin_ravi/models/yolov8s_308_461_epoch1000.onnx"

    # Inference on single image
    if singleImage:
        img, inferenceTime = main(imageFile, classesFile, modelWeights)
        
        if showImage:
            inferenceTime = 'Inference time: %.2f ms' % (inferenceTime)
            cv2.putText(img, inferenceTime, (20, 40), FONT_FACE, FONT_SCALE,
                        (0, 0, 255), THICKNESS, cv2.LINE_AA)
            print(inferenceTime)

            # Show the image.
            cv2.imshow("Prediction", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("Done")

    # inference on multiple images
    else:
        # get all images in the directory
        imageFiles = []
        inferenceTimes = []
        try:
            imageFiles = [os.path.join(imageDir, f) for f in os.listdir(
                imageDir) if os.path.isfile(os.path.join(imageDir, f))]
        except OSError as e:
            print(e)
            exit()

        if createVideo:
            frame = cv2.imread(imageFiles[0])
            height, width, layers = frame.shape
            # create the video writer
            video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))
            inferenceTimes = []

        for imageFile in tqdm(imageFiles):
            img, inferenceTime = main(imageFile, classesFile, modelWeights)
            
            if createVideo:
                # don't show the image
                inferenceTimes.append(inferenceTime)
                inferenceTime = 'Inference time: %.2f ms' % (inferenceTime)
                cv2.putText(img, inferenceTime, (20, 40), FONT_FACE, FONT_SCALE,
                            (0, 0, 255), THICKNESS, cv2.LINE_AA)

                video.write(img)
                
            elif showImage:
                inferenceTimes.append(inferenceTime)
                inferenceTime = 'Inference time: %.2f ms' % (inferenceTime)
                cv2.putText(img, inferenceTime, (20, 40), FONT_FACE, FONT_SCALE,
                            (0, 0, 255), THICKNESS, cv2.LINE_AA)
                print(inferenceTime)
                
                # Show the image.
                cv2.imshow("Prediction", img)
                cv2.waitKey(0)

            else:
                inferenceTimes.append(inferenceTime)
                inferenceTime = 'Inference time: %.2f ms' % (inferenceTime)
                print(inferenceTime)
                
        if createVideo:
            video.release()

        print("\nAverage inference time: ", np.mean(inferenceTimes))
        print("\nDone")
