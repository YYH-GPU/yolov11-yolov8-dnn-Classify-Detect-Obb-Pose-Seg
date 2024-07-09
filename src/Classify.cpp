//
// Created by yyhao on 24-6-30.
//
#include "../include/Classify.h"

Classify::Classify(const std::string& modelPath)
    : YOLO(modelPath)
{
    init_class_name();
}

void Classify::process(const cv::Mat& frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    auto* data = (float*) outs[0].data;
    int dataSize = 1000;
    // 使用 vector 来存储索引，方便排序
    std::vector<int> indices(dataSize);
    for (int i = 0; i < dataSize; ++i)
    {
        indices[i] = i;// 将索引 0 到 dataSize-1 存入 vector
    }

    // 按照 data 中的数值排序 indices
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return data[a] > data[b];// 降序排序
    });

    // 取前五个最大值及其索引存入哈希表
    int count = 0;
    for (int i = 0; i < std::min(5, dataSize); ++i)
    {
        int index = indices[i];
        float value = data[index];
        std::string name = class_name[index];
        out_result.push_back(std::make_pair(name, value));
        count++;
        if (count == 5) break;
    }
    // 使用lambda表达式定义排序规则，根据value进行排序
    std::sort(out_result.begin(), out_result.end(),
              [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
                  return a.second > b.second;// 升序排序
                  // 如果需要降序排序，可以改为 return a.second > b.second;
              });
    // 遍历排序后的vector
    for (const auto& pair: out_result)
    {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    }
}

void Classify::draw(const cv::Mat& frame)
{
    int baseLine;
    int lineHeight = 20;// 每行文字的高度（你可以根据需要调整）
    int textOffset = 10;// 文字距离图片左侧的偏移量（你也可以根据需要调整）

    // 遍历数据，并将每一行文字绘制到图片上
    for (size_t i = 0; i < out_result.size(); ++i)
    {
        std::stringstream ss;
        ss << out_result[i].first << ": " << out_result[i].second;// 将key和value转换为字符串
        std::string text = ss.str();

        // 使用putText函数绘制文字
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, text, cv::Point(textOffset, i * lineHeight + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}
void Classify::init_class_name()
{
    std::ifstream file("/home/yyhao/yolo_dnn/data/cfg/className_1000.txt");// 假设你的txt文件名为data.txt
    std::string line;

    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            // 查找并提取单引号内的字符串
            size_t startPos = line.find('\'');
            while (startPos != std::string::npos)
            {
                size_t endPos = line.find('\'', startPos + 1);
                if (endPos != std::string::npos)
                {
                    // 提取并添加到vector中
                    class_name.push_back(line.substr(startPos + 1, endPos - startPos - 1));
                    // 跳过已处理的字符串和结束的单引号
                    startPos = endPos + 1;
                }
                else
                {
                    // 如果找不到结束的单引号，可能是文件末尾或格式错误，直接跳出循环
                    break;
                }
                // 查找下一个单引号（如果有的话）
                startPos = line.find('\'', startPos + 1);
            }
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file";
        return;
    }
    // 输出结果
//    for (const auto& str : class_name) {
//        std::cout << str << std::endl;
//    }
//    std::cout << class_name.size() << std::endl;
}
