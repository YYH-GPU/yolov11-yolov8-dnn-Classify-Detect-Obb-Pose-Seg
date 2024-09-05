//
// Created by yyhao on 24-6-30.
//

#include "../include/Pose.h"


Pose::Pose(const std::string &modelPath)
    : YOLO(modelPath)
{
}

void Pose::yolov8_Pose_process(cv::Mat &out, int img_w, std::vector<cv::Rect> &bboxes, std::vector<float> &scores, std::vector<int> &classes, std::vector<keypoint> &keyPoint)
{
    int stride = img_w / out.size[2];
    int classNum = out.size[1] - 64 - 51;
    float recore_keypoint[51] = {0.0};
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
                for (int i = 0; i < 51; i++)
                {
                    recore_keypoint[i] = data[(65 + i) * gridX * gridY + shift_num];
                }
                keypoint tempPose;
                for (int i = 0; i < 17; i++)
                {
                    float point_x = ((recore_keypoint[i * 3] * 2.0f) + shiftX) * stride;
                    float point_y = ((recore_keypoint[i * 3 + 1] * 2.0f) + shiftY) * stride;
                    float visible = sigmoid(recore_keypoint[i * 3 + 2]);
                    tempPose.point[i] = restoreCoordinates(cv::Point2f(point_x, point_y), int(org_Size.width), int(org_Size.height), 640, 640);
                    tempPose.visibility[i] = visible;
                }
                keyPoint.push_back(tempPose);

                float x1 = (-record[0] + 0.5f + shiftX) * stride;
                float y1 = (-record[1] + 0.5f + shiftY) * stride;
                float x2 = (record[2] + 0.5f + shiftX) * stride;
                float y2 = (record[3] + 0.5f + shiftY) * stride;

                cv::Rect box;
                box.x = std::max(int(x1), 0);
                box.y = std::max(int(y1), 0);

                box.width = std::max(int(x2 - x1), 0);
                box.height = std::max(int(y2 - y1), 0);
                box = restoreRect(box, org_Size.width, org_Size.height, 640, 640);
                bboxes.push_back(box);
                scores.push_back(max_score);
                classes.push_back(cid);
            }
        }
    }
}

void Pose::detect_Pose(const cv::Mat &frame, std::vector<YOLO_OUT_POSE> &yoloOut)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    auto start = std::chrono::high_resolution_clock::now();
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算运行时间，单位为毫秒
    std::chrono::duration<double, std::milli> duration = end - start;

    // 输出运行时间
    std::cout << "代码运行时间: " << duration.count() << " 毫秒" << std::endl;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<keypoint> keyPoint;
    int img_w = frame.size[1];// 使用宽度
    for (int i = 0; i < outs.size(); i++)
    {
        yolov8_Pose_process(outs[i], img_w, bboxes, scores, classes, keyPoint);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, confThreshold, nmsThreshold, indices);
    //    std::cout << "indices.size: " << indices.size() << std::endl;
    for (int i = 0; i < indices.size(); i++)
    {
        YOLO_OUT_POSE yoloOutTemp;
        const int idx = indices[i];
        yoloOutTemp.outRect = bboxes[idx];
        yoloOutTemp.score = scores[idx];
        yoloOutTemp.classId = classes[idx];
        for (int j = 0; j < 17; j++)
        {
            yoloOutTemp.key_Point.point[j] = keyPoint[idx].point[j];
            yoloOutTemp.key_Point.visibility[j] = keyPoint[idx].visibility[j];
        }
        yoloOut.push_back(yoloOutTemp);
    }
}

void Pose::draw_Pose(cv::Mat &image, std::vector<YOLO_OUT_POSE> yolo_out)
{
    // 定义连接线的索引
    std::vector<std::vector<int>> lines = {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}};

    for (const auto &yolo_out_pose: yolo_out)
    {
        // 绘制检测框
        cv::rectangle(image, yolo_out_pose.outRect, random_color(yolo_out_pose.classId), 2);

        // 绘制类别和置信度文本
        std::string text = classNames_coco80[yolo_out_pose.classId] + " " + std::to_string(yolo_out_pose.score).substr(0, 4);
        cv::Point textOrg(yolo_out_pose.outRect.x, yolo_out_pose.outRect.y - 8);// 文本位置偏移
        cv::putText(image, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, random_color(yolo_out_pose.classId), 2, cv::LINE_AA);

        // 绘制关键点和连接线
        for (const auto &line_pair: lines)
        {
            int idx1 = line_pair[0] - 1;
            int idx2 = line_pair[1] - 1;

            // 检查关键点可见性
            if (yolo_out_pose.key_Point.visibility[idx1] >= 0.5 && yolo_out_pose.key_Point.visibility[idx2] >= 0.5)
            {
                cv::Point pos1 = yolo_out_pose.key_Point.point[idx1];
                cv::Point pos2 = yolo_out_pose.key_Point.point[idx2];

                // 检查关键点位置有效性
                if (pos1.x != 0 && pos1.y != 0 && pos2.x != 0 && pos2.y != 0)
                {
                    // 绘制连接线
                    cv::line(image, pos1, pos2, limb_color[line_pair[0]], 2, cv::LINE_AA);

                    // 绘制关键点
                    cv::circle(image, pos1, 5, kpt_color[idx1], cv::FILLED, cv::LINE_AA);
                    cv::circle(image, pos2, 5, kpt_color[idx2], cv::FILLED, cv::LINE_AA);
                }
            }
        }
    }
}
