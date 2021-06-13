#pragma once

#include <vector>
#include <array>
#include <opencv4/opencv2/opencv.hpp>

/*
class Kmeans
{
public:
    Kmeans(size_t k);

    void fit();
    std::vector<> predict(cv::Mat lbp);

private:
    size_t k_;
};
 */

std::pair<cv::Mat, cv::Mat> kmeans(size_t k, cv::Mat lbp);
