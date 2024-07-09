#ifndef YOLO_TEST_H
#define YOLO_TEST_H
#include "Classify.h"
#include "Detect.h"
#include "Obb.h"
#include "Pose.h"
#include "Seg.h"
#include "YOLO.h"

class TEST
{
public:
    TEST(const std::string& imagePath);
    TEST();
    virtual ~TEST() = default;

protected:
    cv::Mat image;
};

class test_Classify : public TEST, public Classify
{
public:
    test_Classify(const std::string& modelPath, const std::string& imagePath);
    test_Classify(const std::string& modelPath);
    void runTest();
    void runCameraTest();
};

class test_Detect : public TEST, public Detect
{
public:
    test_Detect(const std::string& modelPath, const std::string& imagePath);
    test_Detect(const std::string& modelPath);
    void runTest();
    void runCameraTest();
};

class test_Pose : public TEST, public Pose
{
public:
    test_Pose(const std::string& modelPath, const std::string& imagePath);
    test_Pose(const std::string& modelPath);
    void runTest();
    void runCameraTest();
};

class test_Seg : public TEST, public Seg
{
public:
    test_Seg(const std::string& modelPath, const std::string& imagePath);
    test_Seg(const std::string& modelPath);
    void runTest();
    void runCameraTest();
};

class test_Obb : public TEST, public Obb
{
public:
    test_Obb(const std::string& modelPath, const std::string& imagePath);
    test_Obb(const std::string& modelPath);
    void runTest();
//    void runCameraTest();
};

#endif// YOLO_TEST_H
