#pragma once

#include <vector>
#include <array>
#include <opencv4/opencv2/opencv.hpp>

std::pair<cv::Mat, cv::Mat> kmeans(size_t k, cv::Mat lbp);
