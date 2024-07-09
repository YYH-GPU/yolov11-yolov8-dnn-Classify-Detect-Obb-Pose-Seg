//
// Created by yyhao on 24-6-28.
//
#include "../include/YOLO.h"

YOLO::YOLO(const std::string& modelPath)
{
    this->modelPath = modelPath;
    loadModel();
}

void YOLO::loadModel()
{
    net = cv::dnn::readNet(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// HSV 转换为 BGR 函数
cv::Scalar YOLO::hsv2bgr(float h, float s, float v)
{
    int h_i = static_cast<int>(h * 6);
    float f = h * 6 - h_i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    float r, g, b;
    switch (h_i)
    {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;// Should never be reached
    }

    return {b * 255, g * 255, r * 255};
}

// 生成随机颜色
cv::Scalar YOLO::random_color(int id)
{
    float h = float((((id << 2) ^ 0x937151) % 100)) / 100.0f;
    float s = float((((id << 3) ^ 0x315793) % 100)) / 100.0f;
    return hsv2bgr(h, s, 1.0f);
}


cv::Mat YOLO::letterbox(const cv::Mat& image, int targetWidth, int targetHeight, cv::Scalar fillValue = cv::Scalar(0, 0, 0))
{
    cv::Mat result(targetHeight, targetWidth, image.type(), fillValue);

    // Calculate aspect ratios
    float widthRatio = static_cast<float>(targetWidth) / image.cols;
    float heightRatio = static_cast<float>(targetHeight) / image.rows;

    // Determine scaling factors while maintaining aspect ratio
    float scaleFactor = std::min(widthRatio, heightRatio);

    // Compute new dimensions
    int newWidth = static_cast<int>(image.cols * scaleFactor);
    int newHeight = static_cast<int>(image.rows * scaleFactor);

    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newWidth, newHeight));

    // Place the resized image in the center of the result image (letterboxing)
    cv::Rect roi((targetWidth - newWidth) / 2, (targetHeight - newHeight) / 2, newWidth, newHeight);
    resized.copyTo(result(roi));

    return result;
}
void YOLO::softmax(float* input, int length)
{
    float maxn = input[0];
    for (int i = 1; i < length; i++)
    {
        if (maxn < input[i]) { maxn = input[i]; }
    }

    float sum = 0.0;
    for (int i = 0; i < length; i++)
    {
        input[i] = std::exp(input[i] - maxn);
        sum += input[i];
    }

    for (int i = 0; i < length; i++)
    {
        input[i] /= sum;
    }
}
float YOLO::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// 函数原型：将经过 letterbox 处理后的 Rect 框还原到原始图像大小
cv::Rect YOLO::restoreRect(const cv::Rect& rect, int originalWidth, int originalHeight, int targetWidth, int targetHeight)
{
    // Calculate aspect ratio used for letterboxing
    float scale = std::min(static_cast<float>(targetWidth) / originalWidth, static_cast<float>(targetHeight) / originalHeight);

    // Calculate offsets due to letterboxing
    int offsetX = (targetWidth - scale * originalWidth) / 2;
    int offsetY = (targetHeight - scale * originalHeight) / 2;

    // Calculate original coordinates
    int x1_original = static_cast<int>((rect.x - offsetX) / scale);
    int y1_original = static_cast<int>((rect.y - offsetY) / scale);
    int x2_original = static_cast<int>((rect.x + rect.width - offsetX) / scale);
    int y2_original = static_cast<int>((rect.y + rect.height - offsetY) / scale);

    // Create the original Rect
    cv::Rect originalRect(x1_original, y1_original, x2_original - x1_original, y2_original - y1_original);

    // Ensure the original Rect stays within the bounds of the original image
    originalRect.x = std::max(originalRect.x, 0);
    originalRect.y = std::max(originalRect.y, 0);
    originalRect.width = std::min(originalRect.width, originalWidth - originalRect.x);
    originalRect.height = std::min(originalRect.height, originalHeight - originalRect.y);

    return originalRect;
}

cv::Point2f YOLO::restoreCoordinates(const cv::Point2f& point, int originalWidth, int originalHeight, int targetWidth, int targetHeight)
{
    // Calculate aspect ratios
    float widthRatio = static_cast<float>(targetWidth) / originalWidth;
    float heightRatio = static_cast<float>(targetHeight) / originalHeight;

    // Determine scaling factors while maintaining aspect ratio
    float scaleFactor = std::min(widthRatio, heightRatio);

    // Compute new dimensions
    int newWidth = static_cast<int>(originalWidth * scaleFactor);
    int newHeight = static_cast<int>(originalHeight * scaleFactor);

    // Compute offsets (letterboxing)
    int offsetX = (targetWidth - newWidth) / 2;
    int offsetY = (targetHeight - newHeight) / 2;

    // Convert point coordinates back to the original scale
    float originalX = (point.x - offsetX) / scaleFactor;
    float originalY = (point.y - offsetY) / scaleFactor;

    return cv::Point2f(originalX, originalY);
}
float YOLO::square(float x)
{
    return x * x;// 返回输入数的平方
}

void YOLO::preprocess_warpAffine(cv::Mat& image, cv::Mat& img_pre, cv::Mat& IM, int dst_width, int dst_height)
{
    // Calculate scaling factor
    float scale = std::min(static_cast<float>(dst_width) / image.cols, static_cast<float>(dst_height) / image.rows);

    // Calculate offsets
    float ox = (dst_width - scale * image.cols) / 2;
    float oy = (dst_height - scale * image.rows) / 2;

    // Build affine transformation matrix
    cv::Mat M = (cv::Mat_<float>(2, 3) << scale, 0, ox, 0, scale, oy);

    // Apply affine transformation
    cv::warpAffine(image, img_pre, M, cv::Size(dst_width, dst_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Compute inverse affine transformation matrix
    cv::invertAffineTransform(M, IM);
}

void YOLO::restore_mask(const cv::Mat& mask, cv::Mat& restored_mask, const cv::Mat& IM, int orig_width, int orig_height)
{
    cv::warpAffine(mask, restored_mask, IM, cv::Size(orig_width, orig_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
}

cv::Rect YOLO::transformRect(const cv::Rect& box, const cv::Mat& IM)
{
    // Convert rectangle corners to points
    cv::Point2f tl(box.x, box.y); // Top-left corner
    cv::Point2f br(box.x + box.width, box.y + box.height); // Bottom-right corner

    // Transform points using inverse affine matrix
    std::vector<cv::Point2f> points{tl, br};
    std::vector<cv::Point2f> transformedPoints;
    cv::transform(points, transformedPoints, IM);

    // Calculate the new bounding box in the original image space
    cv::Rect transformedRect(transformedPoints[0], transformedPoints[1]);

    return transformedRect;
}
