#pragma once

#include <opencv4/opencv2/opencv.hpp>

cv::Mat nearest_neighbour(const cv::Mat &lbp, const cv::Mat &centroids);
