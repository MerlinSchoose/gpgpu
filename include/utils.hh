#pragma once

#include <vector>
#include <opencv4/opencv2/opencv.hpp>

std::vector<unsigned char> mat_to_vect(cv::Mat image);
cv::Mat vect_to_mat(std::vector<unsigned char> image);
