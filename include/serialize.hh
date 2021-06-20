#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <string>

void deserializeMat(cv::Mat& mat, const std::string& filename);
void serializeMat(const std::string& filename, cv::Mat& mat);
