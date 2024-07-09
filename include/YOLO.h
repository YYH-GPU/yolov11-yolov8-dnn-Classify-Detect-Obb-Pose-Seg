//
// Created by yyhao on 24-6-28.
//

#ifndef YOLO_DNN_YOLO_H
#define YOLO_DNN_YOLO_H

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

class YOLO
{
public:
    YOLO(const std::string& modelPath);
    virtual ~YOLO() = default;
    void loadModel();
    cv::Scalar hsv2bgr(float h, float s, float v);
    cv::Scalar random_color(int id);
    cv::Mat letterbox(const cv::Mat& image, int targetWidth, int targetHeight, cv::Scalar fillValue);
    cv::Rect restoreRect(const cv::Rect& rect, int originalWidth, int originalHeight, int targetWidth, int targetHeight);
    cv::Point2f restoreCoordinates(const cv::Point2f& point, int originalWidth, int originalHeight, int targetWidth, int targetHeight);
    void preprocess_warpAffine(cv::Mat& image, cv::Mat& img_pre, cv::Mat& IM, int dst_width = 640, int dst_height = 640);
    void restore_mask(const cv::Mat& mask, cv::Mat& restored_mask, const cv::Mat& IM, int orig_width, int orig_height);
    cv::Rect transformRect(const cv::Rect& box, const cv::Mat& IM);

    void softmax(float* input, int length);
    float sigmoid(float x);
    float square(float x);

protected:
    std::string modelPath;
    cv::dnn::Net net;
    float confThreshold = 0.25;
    float nmsThreshold = 0.45;
    std::vector<std::string> classNames_coco80 = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"};
};
#endif// YOLO_DNN_YOLO_H
