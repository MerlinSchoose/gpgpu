#pragma once

#include <opencv4/opencv2/opencv.hpp>

cv::Mat render(const cv::Mat &image,const cv::Mat &histos, unsigned char *colors);
