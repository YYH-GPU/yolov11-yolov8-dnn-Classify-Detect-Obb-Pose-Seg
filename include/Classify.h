//
// Created by yyhao on 24-6-30.
//

#ifndef YOLO_CLASSIFY_H
#define YOLO_CLASSIFY_H
#include "YOLO.h"

class Classify : public YOLO
{
public:
    explicit Classify(const std::string& modelPath);
    void process(const cv::Mat& frame);
    void draw(const cv::Mat& frame);
    void init_class_name();
    std::vector<std::pair<std::string, float>> out_result;
private:
    std::vector<std::string> class_name;
};


#endif//YOLO_CLASSIFY_H
