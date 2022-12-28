/*
 * Copyright 2022 Bonn-Rhein-Sieg University
 *
 * Author: Kevin Patel
 *
 */

#include <yolo_inference.hpp>

YoloInference::YoloInference(std::string net_file, std::string classes_file)
{
    // Read class names from file.
    ifstream ifs(classes_file);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    // Load the network.
    net = readNet(net_file);
}

YoloInference::~YoloInference()
{}

RecognizedObjectList YoloInference::run_inference(Mat &input_image)
{
    // Inference on the image.
    vector<Mat> detections = pre_process(input_image);
    
    // Run NMS
    RecognizedObjectList recoglist = post_process(input_image, detections);

    return recoglist;
}

void YoloInference::draw_label(Mat &input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> YoloInference::pre_process(Mat &input_image)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

RecognizedObjectList YoloInference::post_process(Mat &input_image, vector<Mat> &outputs)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    RecognizedObjectList recognized_object_list;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    // dimensions = [num. of classes] + [x, y, w, h, confidence]
    const int dimensions = class_list.size() + 5.0;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_list.size(), CV_32FC1, classes_scores);

            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += dimensions;
    }

    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int i=0; i < indices.size(); i++)
    {
        int idx = indices[i];
        RecognizedObject obj;
        Rect box = boxes[idx];
        obj.roi = {box.x, box.y, box.width, box.height};
        string cls_name = class_list[class_ids[idx]];
        std::transform(cls_name.begin(), cls_name.end(), cls_name.begin(), ::toupper);
        obj.class_name = cls_name;
        obj.confidence = confidences[idx];
        recognized_object_list.push_back(obj);
    }

    return recognized_object_list;
}

int main(int argc, char ** argv)
{
    // Image file
    string imageFile = "../images/zidane.jpg";

    // Class names file
    string classFile = "../config/coco.names";

    // Model weights file
    string modelWeights = "../weights/yolov5m.onnx";

    // check if image file exists
    ifstream f1(imageFile.c_str());
    if (!f1.good())
    {
        cout << "Image file not found" << endl;
        return 0;
    }

    // check if class names file exists
    ifstream f2(classFile.c_str());
    if (!f2.good())
    {
        cout << "Class names file not found" << endl;
        return 0;
    }

    // check if model weights file exists
    ifstream f3(modelWeights.c_str());
    if (!f3.good())
    {
        cout << "Model weights file not found" << endl;
        return 0;
    }

    Mat frame;
    frame = cv::imread(imageFile);

    // initialize the class
    YoloInference yolo_inference(modelWeights, classFile);
    
    RecognizedObjectList recoglist = yolo_inference.run_inference(frame);

    // Draw bounding boxes and class labels.
    for (int i = 0; i < recoglist.size(); i++)
    {
        RecognizedObject obj = recoglist[i];
        int left = obj.roi[0];
        int top = obj.roi[1];
        int width = obj.roi[2];
        int height = obj.roi[3];
        string class_name = obj.class_name;
        float conf = obj.confidence;
        // Draw bounding box.
        rectangle(frame, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", conf);
        label = class_name + ":" + label;
        // Draw class labels.
        yolo_inference.draw_label(frame, label, left, top);
    }

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes).
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = yolo_inference.net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(frame, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    imshow("Output", frame);
    waitKey(0);
    return 0;
}