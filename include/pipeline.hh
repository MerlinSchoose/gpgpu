#pragma once

#include <opencv4/opencv2/opencv.hpp>


cv::Mat pipepline_cpu(const cv::Mat &image, unsigned char *colors);

cv::Mat pipepline_gpu(const cv::Mat &image, unsigned char *colors);

cv::Mat pipepline_gpu_opti(const cv::Mat &image, unsigned char *colors);
