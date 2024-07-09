//
// Created by yyhao on 24-6-30.
//

#ifndef YOLO_POSE_H
#define YOLO_POSE_H
#include "YOLO.h"

struct keypoint {
    cv::Point point[17];
    float visibility[17];
};

struct YOLO_OUT_POSE {
    cv::Rect outRect;
    keypoint key_Point;
    int classId;
    float score;
};

class Pose : public YOLO
{
public:
    explicit Pose(const std::string& modelPath);
    void detect_Pose(const cv::Mat& frame, std::vector<YOLO_OUT_POSE>& yoloOut);
    void yolov8_Pose_process(cv::Mat& out, int img_w, std::vector<cv::Rect>& bboxes, std::vector<float>& scores, std::vector<int>& classes, std::vector<keypoint>& keyPoint);
    void draw_Pose(cv::Mat& images, std::vector<YOLO_OUT_POSE> yolo_out);
    cv::Size org_Size;
    // 定义颜色调色板
    std::vector<cv::Vec3b> pose_palette = {
            {255, 128, 0},
            {255, 153, 51},
            {255, 178, 102},
            {230, 230, 0},
            {255, 153, 255},
            {153, 204, 255},
            {255, 102, 255},
            {255, 51, 255},
            {102, 178, 255},
            {51, 153, 255},
            {255, 153, 153},
            {255, 102, 102},
            {255, 51, 51},
            {153, 255, 153},
            {102, 255, 102},
            {51, 255, 51},
            {0, 255, 0},
            {0, 0, 255},
            {255, 0, 0},
            {255, 255, 255}};

    // 定义关键点颜色
    std::vector<cv::Vec3b> kpt_color = {
            pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16],
            pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0],
            pose_palette[0], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9],
            pose_palette[9], pose_palette[9]};

    // 定义肢体颜色
    std::vector<cv::Vec3b> limb_color = {
            pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[7],
            pose_palette[7], pose_palette[7], pose_palette[0], pose_palette[0], pose_palette[0],
            pose_palette[0], pose_palette[0], pose_palette[16], pose_palette[16], pose_palette[16],
            pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16]};
};


#endif//YOLO_POSE_H
