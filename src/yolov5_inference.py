import yolov5
import time
import cv2


start = time.time()
# or load custom model
model = yolov5.load('best_nano_2.pt')
end = time.time()
print("Model load time (ms): ", (end - start) * 1000)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img1 = 'frame0000.jpg'
img1 = cv2.imread(img1)

# perform inference
results = model(img1)
results.show()