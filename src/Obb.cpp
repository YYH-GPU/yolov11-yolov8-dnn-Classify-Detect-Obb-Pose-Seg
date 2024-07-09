//
// Created by yyhao on 24-7-1.
//

#include "Obb.h"

Obb::Obb(const std::string &modelPath)
    : YOLO(modelPath)
{
}

void Obb::detect_Obb(const cv::Mat &frame, std::vector<YOLO_OUT_OBB> &yoloOut)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(1024, 1024), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<std::vector<cv::Point>> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<float> angles;
    int img_w = frame.size[1];// 使用宽度
    for (int i = 0; i < outs.size(); i++)
    {
        yolov8_Obb_process(outs[i], img_w, yoloOut);
    }
    probiou_nms(yoloOut, nmsThreshold);
    for (int i = 0; i < yoloOut.size(); i++)
    {
        float w = yoloOut[i].point_Obb[1].x - yoloOut[i].point_Obb[0].x;
        float h = yoloOut[i].point_Obb[1].y - yoloOut[i].point_Obb[0].y;
        float x_c = yoloOut[i].point_Obb[0].x + w / 2;
        float y_c = yoloOut[i].point_Obb[0].y + h / 2;
        float theta = yoloOut[i].angle;
        std::vector<cv::Point> out = get_rotated_bbox(x_c, y_c, w, h, theta);
        yoloOut[i].out_Point_Obb = out;
    }
}

void Obb::draw_Obb(cv::Mat &image, std::vector<YOLO_OUT_OBB> yolo_out)
{
    for (const auto &det: yolo_out)
    {
        for (const auto &det: yolo_out)
        {
            // 绘制 OBB 多边形
            cv::polylines(image, det.out_Point_Obb, true, random_color(det.classId), 2);

            // 获取标签和置信度文本
            std::string caption = class_names_obb[det.classId] + " " + std::to_string(det.score).substr(0, 4);

            // 计算标签文本框的位置和大小
            int text_baseline;
            cv::Size textSize = cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &text_baseline);
            int text_width = textSize.width + 10;
            int text_height = textSize.height + 10;

            // 获取 OBB 左下角点
            cv::Point bottom_left = *std::min_element(det.out_Point_Obb.begin(), det.out_Point_Obb.end(),
                                                      [](const cv::Point &a, const cv::Point &b) {
                                                          return (a.y == b.y) ? a.x < b.x : a.y > b.y;
                                                      });

            // 确保文本框不超出图像边界
            int left = bottom_left.x;

            // 绘制标签背景框
            cv::rectangle(image, cv::Point(left, bottom_left.y - text_height), cv::Point(left + text_width, bottom_left.y), random_color(det.classId), -1);

            // 绘制标签文本
            cv::putText(image, caption, cv::Point(left + 5, bottom_left.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2, 16);
        }
    }
}

void Obb::yolov8_Obb_process(cv::Mat &out, int img_w, std::vector<YOLO_OUT_OBB> &yoloOut)
{
    {
        int stride = img_w / out.size[2];
        int classNum = out.size[1] - 64 - 1;
        int gridX = out.size[2];
        int gridY = gridX;

        auto *data = (float *) out.data;

        for (int shiftY = 0; shiftY < gridY; shiftY++)
        {
            for (int shiftX = 0; shiftX < gridX; shiftX++)
            {
                int shift_num = shiftX + shiftY * gridX;
                const float *cls_ptr = data + 64 * gridY * gridX + shift_num;

                float max_score = -1.0f;
                int cid = -1;
                for (int j = 0; j < classNum; j++)
                {
                    float score = sigmoid(cls_ptr[j * gridY * gridX]);
                    if (score > max_score)
                    {
                        max_score = score;
                        cid = j;
                    }
                }

                if (max_score > confThreshold)
                {
                    float record[4] = {0};
                    for (int i = 0; i < 4; i++)
                    {
                        float recordvector[16];
                        for (int j = 0; j < 16; j++)
                        {
                            recordvector[j] = data[(i * 16 + j) * gridY * gridX + shift_num];
                        }
                        softmax(recordvector, 16);
                        for (int j = 0; j < 16; j++)
                        {
                            record[i] += j * recordvector[j];
                        }
                    }
                    float angle = (sigmoid(*(data + (64 + classNum) * gridY * gridX + shift_num)) - 0.25) * M_PI;
                    float x1 = (-record[0] + 0.5f + shiftX) * stride;
                    float y1 = (-record[1] + 0.5f + shiftY) * stride;
                    float x2 = (record[2] + 0.5f + shiftX) * stride;
                    float y2 = (record[3] + 0.5f + shiftY) * stride;
                    cv::Point topLeft = restoreCoordinates(cv::Point2f(x1, y1), org_Size.width, org_Size.height, 1024, 1024);
                    cv::Point bottomRight = restoreCoordinates(cv::Point2f(x2, y2), org_Size.width, org_Size.height, 1024, 1024);
                    YOLO_OUT_OBB temp_Obb;
                    temp_Obb.point_Obb.push_back(topLeft);
                    temp_Obb.point_Obb.push_back(bottomRight);
                    temp_Obb.classId = cid;
                    temp_Obb.angle = angle;
                    temp_Obb.score = max_score;
                    yoloOut.push_back(temp_Obb);
                }
            }
        }
    }
}

void Obb::covariance_matrix(float w, float h, float r, float &a_val, float &b_val, float &c_val)
{
    // 检查输入参数的有效性
    if (w < 0 || h < 0)
    {
        a_val = b_val = c_val = 0;
        return;
    }
    float a = (w * w) / 12;
    float b = (h * h) / 12;
    float cos_r = cosf(r);
    float sin_r = sinf(r);

    a_val = a * cos_r * cos_r + b * sin_r * sin_r;
    b_val = a * sin_r * sin_r + b * cos_r * cos_r;
    c_val = (a - b) * sin_r * cos_r;
}

float Obb::probiou(const YOLO_OUT_OBB &obb1, const YOLO_OUT_OBB &obb2, float eps)
{
    float a1, b1, c1;
    float a2, b2, c2;
    covariance_matrix(obb1.point_Obb[1].x - obb1.point_Obb[0].x, obb1.point_Obb[1].y - obb1.point_Obb[0].y, obb1.angle, a1, b1, c1);
    covariance_matrix(obb2.point_Obb[1].x - obb2.point_Obb[0].x, obb2.point_Obb[1].y - obb2.point_Obb[0].y, obb2.angle, a2, b2, c2);

    float x1 = obb1.point_Obb[0].x + (obb1.point_Obb[1].x - obb1.point_Obb[0].x) / 2;
    float y1 = obb1.point_Obb[0].y + (obb1.point_Obb[1].y - obb1.point_Obb[0].y) / 2;
    float x2 = obb2.point_Obb[0].x + (obb2.point_Obb[1].x - obb2.point_Obb[0].x) / 2;
    float y2 = obb2.point_Obb[0].y + (obb2.point_Obb[1].y - obb2.point_Obb[0].y) / 2;

    float t1 = ((a1 + a2) * square(y1 - y2) + (b1 + b2) * square(x1 - x2)) / ((a1 + a2) * (b1 + b2) - square(c1 + c2) + eps);
    float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - square(c1 + c2) + eps);
    float t3 = log(((a1 + a2) * (b1 + b2) - square(c1 + c2)) / (4 * sqrt(a1 * b1 - square(c1)) * sqrt(a2 * b2 - square(c2)) + eps) + eps);

    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    float hd = sqrtf(1.0f - expf(-fminf(fmaxf(bd, eps), 100.0f)) + eps);

    return 1.0f - hd;
}

void Obb::boxScoreSort(std::vector<YOLO_OUT_OBB> &yoloOut)
{
    std::sort(yoloOut.begin(), yoloOut.end(), [](const YOLO_OUT_OBB &a, const YOLO_OUT_OBB &b) {
        return a.score > b.score;
    });
}

void Obb::probiou_nms(std::vector<YOLO_OUT_OBB> &yoloOut, float nmsThresh)
{
    boxScoreSort(yoloOut);

    std::vector<bool> remove_flags(yoloOut.size(), false);
    for (size_t i = 0; i < yoloOut.size(); ++i)
    {
        if (remove_flags[i] || yoloOut[i].score == 0) continue;

        for (size_t j = i + 1; j < yoloOut.size(); ++j)
        {
            if (remove_flags[j] || yoloOut[j].score == 0 || yoloOut[i].classId != yoloOut[j].classId) continue;

            if (probiou(yoloOut[i], yoloOut[j], 1e-7) > nmsThresh)
            {
                remove_flags[j] = true;
            }
        }
    }

    std::vector<YOLO_OUT_OBB> newBoxes;
    for (size_t i = 0; i < yoloOut.size(); ++i)
    {
        if (!remove_flags[i])
        {
            newBoxes.push_back(yoloOut[i]);
        }
    }
    yoloOut = std::move(newBoxes);
}

cv::Point Obb::rotate_point(float x, float y, float theta)
{
    float x_new = x * cos(theta) - y * sin(theta);
    float y_new = x * sin(theta) + y * cos(theta);
    return cv::Point2f(x_new, y_new);
}

std::vector<cv::Point> Obb::get_rotated_bbox(float x_c, float y_c, float w, float h, float theta)
{
    // 四个顶点相对于中心点的初始坐标
    std::vector<cv::Point2f> corners = {
            cv::Point2f(-w / 2, -h / 2),
            cv::Point2f(w / 2, -h / 2),
            cv::Point2f(w / 2, h / 2),
            cv::Point2f(-w / 2, h / 2)};

    std::vector<cv::Point> rotated_corners;
    for (const auto &corner: corners)
    {
        // 旋转顶点坐标
        cv::Point2f rotated = rotate_point(corner.x, corner.y, theta);
        // 平移回中心点
        rotated.x += x_c;
        rotated.y += y_c;
        rotated_corners.push_back(rotated);
    }

    return rotated_corners;
}
