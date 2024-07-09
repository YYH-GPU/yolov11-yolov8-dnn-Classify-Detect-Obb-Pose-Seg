// Seg.h
//
// Created by yyhao on 24-7-4.
//

#ifndef YOLO_SEG_H
#define YOLO_SEG_H

#include "YOLO.h"

struct YOLO_OUT_SEG {
    cv::Rect outRect;
    int classId;
    float score;
    std::vector<float> seg_weight;
    cv::Mat mask;
};

class Seg : public YOLO
{
public:
    explicit Seg(const std::string& modelPath);
    void detect_Seg(const cv::Mat& frame, std::vector<YOLO_OUT_SEG>& yoloOut);
    void yolov8_Seg_process(cv::Mat& out, int img_w, std::vector<cv::Rect>& bboxes, std::vector<float>& scores,std::vector<int>& classes, std::vector<std::vector<float>>& seg_w);
    void draw_Seg(cv::Mat& images, std::vector<YOLO_OUT_SEG>& yolo_out);
    void draw_rectangle(cv::Mat &image, YOLO_OUT_SEG &yoloout);
    void seg_process(std::vector<cv::Mat> &channels, YOLO_OUT_SEG &yoloOut);
    void draw_segment(cv::Mat &image, cv::Mat &mask, cv::Scalar color);
    cv::Size org_Size;
    cv::Mat IM;
};

#endif // YOLO_SEG_H
