// Seg.cpp
//
// Created by yyhao on 24-7-4.
//

#include "../include/Seg.h"

#include <opencv2/dnn.hpp>

Seg::Seg(const std::string &modelPath)
    : YOLO(modelPath)
{
}

void Seg::yolov8_Seg_process(cv::Mat &out, int img_w, std::vector<cv::Rect> &bboxes, std::vector<float> &scores, std::vector<int> &classes, std::vector<std::vector<float>> &seg_w)
{
    int stride = img_w / out.size[2];
    int classNum = out.size[1] - 64 - 32;
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
                std::vector<float> seg(32);
                for (int i = 0; i < 32; i++)
                {
                    seg[i] = data[(144 + i) * gridX * gridY + shift_num];
                }
                float x1 = (-record[0] + 0.5f + shiftX) * stride;
                float y1 = (-record[1] + 0.5f + shiftY) * stride;
                float x2 = (record[2] + 0.5f + shiftX) * stride;
                float y2 = (record[3] + 0.5f + shiftY) * stride;

                cv::Rect box;
                box.x = std::max(int(x1), 0);
                box.y = std::max(int(y1), 0);
                box.width = std::max(int(x2 - x1), 0);
                box.height = std::max(int(y2 - y1), 0);
                bboxes.push_back(box);
                scores.push_back(max_score);
                classes.push_back(cid);
                seg_w.push_back(seg);
            }
        }
    }
}

void Seg::seg_process(std::vector<cv::Mat> &channels, YOLO_OUT_SEG &yoloOut)
{
    // Initialize result matrix
    cv::Mat result = cv::Mat::zeros(160, 160, CV_32F);

    // Accumulate weighted channels into result
    for (int i = 0; i < 32; i++)
    {
        cv::Mat temp;
        channels[i].convertTo(temp, CV_32F);
        result += temp * yoloOut.seg_weight[i];
    }

    // Resize result to 640x640
    cv::resize(result, result, cv::Size(640, 640));

    // Apply ROI to the result and threshold to 0 or 1
    cv::Rect roi = yoloOut.outRect;
    for (int y = 0; y < result.rows; y++)
    {
        for (int x = 0; x < result.cols; x++)
        {
            if (roi.contains(cv::Point(x, y)))
            {
                result.at<float>(y, x) = (result.at<float>(y, x) > 0.5) ? 1.0f : 0.0f;
            }
            else
            {
                result.at<float>(y, x) = 0.0f;
            }
        }
    }
    result.convertTo(result, CV_8U, 255);
    restore_mask(result, result, IM, org_Size.width, org_Size.height);
    yoloOut.mask = result;
}

void Seg::detect_Seg(const cv::Mat &frame, std::vector<YOLO_OUT_SEG> &yoloOut)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<std::vector<float>> seg_w;
    int img_w = frame.size[1];// Use width
    for (int i = 1; i < 4; i++)
    {
        yolov8_Seg_process(outs[i], img_w, bboxes, scores, classes, seg_w);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, confThreshold, nmsThreshold, indices);

    for (int idx: indices)
    {
        YOLO_OUT_SEG seg;
        seg.outRect = bboxes[idx];
        seg.classId = classes[idx];
        seg.score = scores[idx];
        seg.seg_weight = seg_w[idx];
        yoloOut.push_back(seg);
    }

    // Extract channels from the first output for segmentation processing
    cv::Mat mat(32, 160 * 160, CV_32F, outs[0].data);
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 32; i++)
    {
        cv::Mat slice = mat.row(i).reshape(1, 160);
        channels.push_back(slice);
    }

    // Process each segmentation output
    for (int i = 0; i < yoloOut.size(); i++)
    {
        seg_process(channels, yoloOut[i]);
        yoloOut[i].outRect = transformRect(yoloOut[i].outRect, IM);
    }
}

void Seg::draw_segment(cv::Mat &image, cv::Mat &mask, cv::Scalar color)
{
    CV_Assert(image.size() == mask.size() && image.channels() == 3 && mask.channels() == 1 && mask.type() == CV_8U);

    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (mask.at<uchar>(y, x) > 0)
            {
                image.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(image.at<cv::Vec3b>(y, x)[0] * 0.5 + color[0] * 0.5);
                image.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(image.at<cv::Vec3b>(y, x)[1] * 0.5 + color[1] * 0.5);
                image.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(image.at<cv::Vec3b>(y, x)[2] * 0.5 + color[2] * 0.5);
            }
        }
    }
}

void Seg::draw_rectangle(cv::Mat &image, YOLO_OUT_SEG &yoloout)
{
    cv::Scalar color = random_color(yoloout.classId);
    int thickness = 2;// Thickness of bounding box

    cv::rectangle(image, yoloout.outRect, color, thickness);

    std::string label = classNames_coco80[yoloout.classId] + " " + std::to_string(yoloout.score).substr(0, 4);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    yoloout.outRect.y = std::max(yoloout.outRect.y, labelSize.height);
    cv::putText(image, label, cv::Point(yoloout.outRect.x, yoloout.outRect.y - labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
}

void Seg::draw_Seg(cv::Mat &image, std::vector<YOLO_OUT_SEG> &yolo_out)
{
    for (auto &out: yolo_out)
    {
        draw_rectangle(image, out);
        draw_segment(image, out.mask, random_color(out.classId));// Green color for mask
    }
}
