import yolov5
import time
import os
import cv2


start = time.time()
# or load custom model
model = yolov5.load('/home/kvnptl/work/b_it_bots/robothon_misc/best_nano_2.pt')
# model = yolov5.load('/home/kvnptl/work/b_it_bots/robothon_misc/best_small.pt')
# model = yolov5.load('/home/kvnptl/work/b_it_bots/robothon_misc/visual_servoing/vision_data/task_4_door_knob/best_nano.pt')
end = time.time()
print("Model load time (ms): ", (end - start) * 1000)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
# img1 = '/home/kvnptl/work/b_it_bots/robothon_misc/dataset_robothon2023/images/train/frame0022.jpg'
# img1 = cv2.imread(img1)

# perform inference
# results = model(img1)
# results.show()


# load images
# imageDir = '/home/kvnptl/work/b_it_bots/robothon_misc/small_size/'
imageDir = '/home/kvnptl/work/b_it_bots/robothon_misc/dataset_robothon2023/images/train/'
# imageDir = '/home/kvnptl/work/b_it_bots/robothon_misc/visual_servoing/vision_data/task_4_door_knob/dataset_robothon2023_door/images/train/'

# get all images in the directory
imageFiles = []
inferenceTimes = []
try:
    imageFiles = [os.path.join(imageDir, f) for f in os.listdir(
        imageDir) if os.path.isfile(os.path.join(imageDir, f))]
except OSError as e:
    print(e)
    exit()
inferenceTimes = []
scores_list = []
for img in imageFiles:
    img1 = cv2.imread(img)
    # inference with larger input size
    start = time.time()
    results = model(img1, size=350) # try 640, 512, 480
    end = time.time()
    print("Inference time (ms): ", (end - start) * 1000)
    inferenceTimes.append((end - start) * 1000)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]

    if len(scores) > 0:
        scores_list.append(scores)
        categories = predictions[:, 5]

        # find the center of the image
        center = (img1.shape[1] / 2, img1.shape[0] / 2)

        # draw vertical line at the center of the image
        cv2.line(img1, (int(center[0]), 0),
                 (int(center[0]), img1.shape[0]), (0, 0, 255), 2)

        # find the center of the bounding box
        center_box = (boxes[0][0] + boxes[0][2]) / \
            2, (boxes[0][1] + boxes[0][3]) / 2
        
        # draw the bounding box
        cv2.rectangle(img1, (int(boxes[0][0]), int(boxes[0][1])), (int(
            boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)

        # show the center of the bounding box on the image
        cv2.circle(img1, (int(center_box[0]), int(
            center_box[1])), 4, (255, 255, 0), 1)

        # find the error in y direction
        error_y = center[0] - int(center_box[0])

        # round the error to 2 decimal places
        # error_y = round(float(error_y.numpy()), 2)

        # put a small background behind the text
        cv2.rectangle(img1, (5, 7), (200, 45), (0, 0, 0), -1)

        # print the error on the image on the top left corner of the image
        cv2.putText(img1, "Error (y): " + str(int(error_y)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # draw the error line from the center of bounding box to the y axis of the image
        cv2.line(img1, (int(center_box[0]), int(center_box[1])), (int(
            center_box[0] + error_y), int(center_box[1])), (0, 255, 0), 2)

        # show the image
        cv2.imwrite("OUTPUT_DEMO/" + img.split('/')[-1], img1)
        cv2.imshow("image", img1)
        cv2.waitKey(0)

        # show detection bounding boxes on image
        # results.show()

        # save results into "results/" folder
        # results.save(save_dir='results1/')
    else:
        print("No detection")

# average inference time
print("Average inference time (ms): ", sum(inferenceTimes) / len(inferenceTimes))

# average scores
print("Average scores: ", sum(scores_list) / len(scores_list))