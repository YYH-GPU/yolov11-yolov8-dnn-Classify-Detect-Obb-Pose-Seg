//
// Created by yyhao on 24-7-8.
//
#include "../include/test.h"

TEST::TEST(const std::string& imagePath)
    : image(cv::imread(imagePath))
{
    if (image.empty())
    {
        std::cerr << "Error loading image: " << imagePath << std::endl;
    }
    else
    {
        std::cout << "Loaded image: " << imagePath << std::endl;
    }
}

TEST::TEST()
{
}

test_Classify::test_Classify(const std::string& modelPath, const std::string& imagePath)
    : TEST(imagePath), Classify(modelPath)
{
    loadModel();
}

test_Classify::test_Classify(const std::string& modelPath)
    : TEST(), Classify(modelPath)
{
    loadModel();
}

void test_Classify::runTest()
{
    if (image.empty())
    {
        std::cerr << "No image loaded to run the test." << std::endl;
        return;
    }

    process(image);
    draw(image);
    cv::imshow("Classification Result", image);
    cv::waitKey(0);
}

void test_Classify::runCameraTest()
{
    cv::VideoCapture cap(0);// 打开默认摄像头
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    while (true)
    {
        out_result.clear();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        process(frame);
        draw(frame);
        cv::imshow("Camera Classification Result", frame);

        if (cv::waitKey(1) >= 0)// 按下任意键退出
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

test_Detect::test_Detect(const std::string& modelPath, const std::string& imagePath)
    : TEST(imagePath), Detect(modelPath)
{
    loadModel();
}

test_Detect::test_Detect(const std::string& modelPath)
    : TEST(), Detect(modelPath)
{
    loadModel();
}

void test_Detect::runTest()
{
    if (image.empty())
    {
        std::cerr << "No image loaded to run the test." << std::endl;
        return;
    }
    model_input_size = cv::Size(640, 640);
    std::vector<YOLO_OUT> yoloOut;
    org_Size = image.size();
    cv::Mat out_frame = letterbox(image, model_input_size.width, model_input_size.height, cv::Scalar(114, 114, 114));
    detect(out_frame, yoloOut);
    draw(image, yoloOut);
    cv::imshow("Detected Image", image);
    cv::waitKey(0);
}

void test_Detect::runCameraTest()
{
    cv::VideoCapture cap(0);// 打开默认摄像头
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    while (true)
    {
        std::vector<YOLO_OUT> yoloOut;
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        org_Size = frame.size();
        cv::Mat out_frame = letterbox(frame, 640, 640, cv::Scalar(114, 114, 114));
        detect(out_frame, yoloOut);
        draw(frame, yoloOut);
        cv::imshow("Camera Detection Result", frame);

        if (cv::waitKey(1) >= 0)// 按下任意键退出
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

test_Pose::test_Pose(const std::string& modelPath, const std::string& imagePath)
    : TEST(imagePath), Pose(modelPath)
{
    loadModel();
}

test_Pose::test_Pose(const std::string& modelPath)
    : TEST(), Pose(modelPath)
{
    loadModel();
}

void test_Pose::runTest()
{
    if (image.empty())
    {
        std::cerr << "No image loaded to run the test." << std::endl;
        return;
    }

    std::vector<YOLO_OUT_POSE> yoloOut;
    org_Size = image.size();
    cv::Mat out_frame = letterbox(image, 640, 640, cv::Scalar(144, 144, 144));
    detect_Pose(out_frame, yoloOut);
    draw_Pose(image, yoloOut);
    cv::imshow("Pose Detection Result", image);
    cv::waitKey(0);
}

void test_Pose::runCameraTest()
{
    cv::VideoCapture cap(0);// 打开默认摄像头
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    while (true)
    {
        std::vector<YOLO_OUT_POSE> yoloOut;
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        org_Size = frame.size();
        cv::Mat out_frame = letterbox(frame, 640, 640, cv::Scalar(144, 144, 144));
        detect_Pose(out_frame, yoloOut);
        draw_Pose(frame, yoloOut);
        cv::imshow("Camera Pose Detection Result", frame);

        if (cv::waitKey(1) >= 0)// 按下任意键退出
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

test_Seg::test_Seg(const std::string& modelPath, const std::string& imagePath)
    : TEST(imagePath), Seg(modelPath)
{
    loadModel();
}

test_Seg::test_Seg(const std::string& modelPath)
    : TEST(), Seg(modelPath)
{
    loadModel();
}

void test_Seg::runTest()
{
    if (image.empty())
    {
        std::cerr << "No image loaded to run the test." << std::endl;
        return;
    }

    std::vector<YOLO_OUT_SEG> yoloOut;
    org_Size = image.size();
    cv::Mat img_pre;
    preprocess_warpAffine(image, img_pre, IM, 640, 640);

    // Detect segmentation
    detect_Seg(img_pre, yoloOut);

    // Draw segmentation
    draw_Seg(image, yoloOut);
    cv::imshow("Pose Detection Result", image);
    cv::waitKey(0);
}

void test_Seg::runCameraTest()
{
    cv::VideoCapture cap(0);// 打开默认摄像头
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    while (true)
    {
        std::vector<YOLO_OUT_SEG> yoloOut;
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        org_Size = frame.size();
        // Preprocess image
        cv::Mat img_pre;
        preprocess_warpAffine(frame, img_pre, IM, 640, 640);

        // Detect segmentation
        detect_Seg(img_pre, yoloOut);

        // Draw segmentation
        draw_Seg(frame, yoloOut);

        cv::imshow("Camera Pose Detection Result", frame);
        if (cv::waitKey(1) >= 0)// 按下任意键退出
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

test_Obb::test_Obb(const std::string& modelPath, const std::string& imagePath)
    : TEST(imagePath), Obb(modelPath)
{
    loadModel();
}

test_Obb::test_Obb(const std::string& modelPath)
    : TEST(), Obb(modelPath)
{
    loadModel();
}

void test_Obb::runTest()
{
    if (image.empty())
    {
        std::cerr << "No image loaded to run the test." << std::endl;
        return;
    }
    std::vector<YOLO_OUT_OBB> yoloOut;
    Obb obb(modelPath);
    obb.org_Size = image.size();
    cv::Mat out_frame = obb.letterbox(image, 1024, 1024, cv::Scalar(144, 144, 144));
    obb.detect_Obb(out_frame, yoloOut);
    obb.draw_Obb(image, yoloOut);
    cv::imshow("Pose Detection Result", image);
    cv::waitKey(0);
}
