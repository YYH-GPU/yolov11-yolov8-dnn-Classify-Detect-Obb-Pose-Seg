//
// Created by yyhao on 24-6-28.
//

#ifndef YOLO_DNN_DETECT_H
#define YOLO_DNN_DETECT_H

#include "YOLO.h"

struct YOLO_OUT {
    cv::Rect outRect;
    int classId;
    float score;
};
class Detect : public YOLO
{
public:
    explicit Detect(const std::string& modelPath);
    void detect(const cv::Mat& frame, std::vector<YOLO_OUT>& yoloOut);
    void yolov8_process(cv::Mat& out, int img_w, std::vector<cv::Rect>& bboxes, std::vector<float>& scores,std::vector<int>& classes);
    void draw(cv::Mat& images, std::vector<YOLO_OUT> yolo_out);
    cv::Size org_Size;
    cv::Size model_input_size;
};
#endif// YOLO_DNN_DETECT_H
