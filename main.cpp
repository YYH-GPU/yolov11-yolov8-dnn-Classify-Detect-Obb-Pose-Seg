//
// Created by yyhao on 24-6-28.
//

#include "../include/test.h"


int main()
{
    std::string model_path = "/home/yyhao/yolo_dnn/data/model/yolov8n.onnx";
    std::string image_path = "/home/yyhao/yolo_dnn/data/images/bus.jpg";
//    test_Classify test(model_path, image_path);
//    test.runTest();
//    test.runCameraTest();
    test_Detect test(model_path, image_path);
//    test.runCameraTest();
    test.runTest();
//    test_Pose test(model_path);
//    test.runTest();
//    test.runCameraTest();
//    test_Seg test(model_path);
//    test.runTest();
//    test.runCameraTest();
//    test_Obb test(model_path, image_path);
//    test.runTest();
}
