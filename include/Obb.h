//
// Created by yyhao on 24-7-1.
//

#ifndef YOLO_OBB_H
#define YOLO_OBB_H
#include "YOLO.h"

struct YOLO_OUT_OBB {
    std::vector<cv::Point> point_Obb;
    std::vector<cv::Point> out_Point_Obb;
    int classId;
    float score;
    float angle;
};
class Obb : public YOLO
{
public:
    explicit Obb(const std::string& modelPath);
    void detect_Obb(const cv::Mat& frame, std::vector<YOLO_OUT_OBB>& yoloOut);
    void yolov8_Obb_process(cv::Mat& out, int img_w, std::vector<YOLO_OUT_OBB>& yoloOut);
    void draw_Obb(cv::Mat& images, std::vector<YOLO_OUT_OBB> yolo_out);
    void covariance_matrix(float w, float h, float r, float& a_val, float& b_val, float& c_val);
    float probiou(const YOLO_OUT_OBB& obb1, const YOLO_OUT_OBB& obb2, float eps);
    void boxScoreSort(std::vector<YOLO_OUT_OBB>& yoloOut);
    void probiou_nms(std::vector<YOLO_OUT_OBB>& yoloOut, float nmsThresh);
    cv::Point rotate_point(float x, float y, float theta);
    std::vector<cv::Point> get_rotated_bbox(float x_c, float y_c, float w, float h, float theta);
    cv::Size org_Size;
    std::vector<std::string> class_names_obb = {
            "Plane",
            "Ship",
            "Storage Tank",
            "Baseball Diamond",
            "Tennis Court",
            "Basketball Court",
            "Ground Track Field",
            "Harbor",
            "Bridge",
            "Large Vehicle",
            "Small Vehicle",
            "Helicopter",
            "Roundabout",
            "Soccer Ball Field",
            "Swimming Pool"};
};

#endif//YOLO_OBB_H
