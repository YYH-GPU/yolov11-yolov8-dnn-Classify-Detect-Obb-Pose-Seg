//
// Created by yyhao on 24-6-28.
//

#include "../include/Detect.h"

Detect::Detect(const std::string& modelPath)
    : YOLO(modelPath)
{
}

//void Detect::yolov8_process(cv::Mat& out, int img_w, std::vector<cv::Rect>& bboxes, std::vector<float>& scores, std::vector<int>& classes)
//{
//    std::cout << "out_size: " << out.size << std::endl;
//    int stride = img_w / out.size[2];
//    std::cout << "stride: " << stride << std::endl;
//    int classNum = out.size[1] - 64;
//    float record[4] = {0, 0, 0, 0};
//    float recordvector[16] = {0.0};
//    int gridX = out.size[2];
//    int gridY = gridX;
//    std::cout << "gridX: " << gridX << std::endl;
//    std::cout << "gridY: " << gridY << std::endl;
//    auto* ptr = (float*) out.data;
//    const float* cls_ptr = nullptr;
//    int channel = out.size[1];
//    float* mat_data[channel];
//    std::cout << "out.channel: " << out.size[1] << std::endl;
//    auto* data_head = (float*) out.data;
//    mat_data[0] = data_head;
//    for (int i = 1; i < channel; i++)
//    {
//        data_head += gridX * gridY;
//        mat_data[i] = data_head;
//    }
//    for (int shiftY = 0; shiftY < gridY; shiftY++)
//    {
//        for (int shiftX = 0; shiftX < gridX; shiftX++)
//        {
//            int shift_num = shiftX + (shiftY * gridX);
//            cls_ptr = ptr + 64 * gridY * gridX + shift_num;
//            float maxsorce = *cls_ptr;
//            int cid = 0;
//            for (int j = 0; j < classNum; j++)
//            {
//                if (maxsorce < *(cls_ptr + j * gridY * gridX))
//                {
//                    maxsorce = *(cls_ptr + j * gridY * gridX);
//                    cid = j;
//                }
//            }
//            float score = sigmoid(maxsorce);
//            if (score > confThreshold)
//            {
//                for (int i = 0; i < 4; i++)
//                {
//                    record[i] = 0;
//                    for (int j = 0; j < 16; j++)
//                    {
//                        recordvector[j] = *(mat_data[i * 16 + j] + shift_num);
//                    }
//                    softmax(recordvector, 16);
//                    for (int j = 0; j < 16; j++)
//                    {
//                        record[i] += j * recordvector[j];
//                    }
//                }
//                cv::Rect box;
//                float x1 = (-record[0] + 0.5f + (float) shiftX) * stride;
//                float y1 = (-record[1] + 0.5f + (float) shiftY) * stride;
//                float x2 = (record[2] + 0.5f + (float) shiftX) * stride;
//                float y2 = (record[3] + 0.5f + (float) shiftY) * stride;
//                box.x = std::max(int(x1), 0);
//                box.y = std::max(int(y1), 0);
//                box.width = std::max(int(x2 - x1), 0);
//                box.height = std::max(int(y2 - y1), 0);
//                bboxes.push_back(box);
//                scores.push_back(score);
//                classes.push_back(cid);
//            }
//        }
//    }
//}

void Detect::yolov8_process(cv::Mat& out, int img_w, std::vector<cv::Rect>& bboxes, std::vector<float>& scores, std::vector<int>& classes)
{
    int stride = img_w / out.size[2];
    int classNum = out.size[1] - 64;
    int gridX = out.size[2];
    int gridY = gridX;

    auto* data = (float*) out.data;

    for (int shiftY = 0; shiftY < gridY; shiftY++)
    {
        for (int shiftX = 0; shiftX < gridX; shiftX++)
        {
            int shift_num = shiftX + shiftY * gridX;
            const float* cls_ptr = data + 64 * gridY * gridX + shift_num;

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
                float x1 = (-record[0] + 0.5f + shiftX) * stride;
                float y1 = (-record[1] + 0.5f + shiftY) * stride;
                float x2 = (record[2] + 0.5f + shiftX) * stride;
                float y2 = (record[3] + 0.5f + shiftY) * stride;

                cv::Rect box;
                box.x = std::max(int(x1), 0);
                box.y = std::max(int(y1), 0);
                box.width = std::max(int(x2 - x1), 0);
                box.height = std::max(int(y2 - y1), 0);
                box = restoreRect(box, org_Size.width, org_Size.height, model_input_size.width, model_input_size.height);
                bboxes.push_back(box);
                scores.push_back(max_score);
                classes.push_back(cid);
            }
        }
    }
}

void Detect::detect(const cv::Mat& frame, std::vector<YOLO_OUT>& yoloOut)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(model_input_size.width, model_input_size.height), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    for(int i = 0; i < outs.size();  i++)
    {
        std::cout << outs[i].size << std::endl;
    }

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;
    int img_w = frame.size[1];// 使用宽度
    for (int i = 0; i < outs.size(); i++)
    {
        yolov8_process(outs[i], img_w, bboxes, scores, classes);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, confThreshold, nmsThreshold, indices);
    //    std::cout << "indices.size: " << indices.size() << std::endl;
    for (int i = 0; i < indices.size(); i++)
    {
        YOLO_OUT yoloOutTemp;
        const int idx = indices[i];
        yoloOutTemp.outRect = bboxes[idx];
        yoloOutTemp.score = scores[idx];
        yoloOutTemp.classId = classes[idx];
        yoloOut.push_back(yoloOutTemp);
    }
}

void Detect::draw(cv::Mat& image, const std::vector<YOLO_OUT> yolo_Out)
{
    int thickness = 3;
    int lineType = 8;
    for (auto out: yolo_Out)
    {
        std::string text = classNames_coco80[out.classId];
        text += " ";
        text += std::to_string(out.score).substr(0, 5);
        cv::rectangle(image, out.outRect, random_color(out.classId), thickness, lineType);
        cv::Point textOrg(out.outRect.x, out.outRect.y - 8);// 假设文本高度为10个像素
        float fontScale = 0.5;
        thickness = 2;
        cv::putText(image, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, fontScale, random_color(out.classId), thickness, 8, false);
    }
}
