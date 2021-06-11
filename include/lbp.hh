#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#define HISTO_SIZE 256

unsigned char get(cv::Mat patch, int i, int j);
unsigned char get_texton(cv::Mat patch, int i, int j);
std::vector<unsigned char> textonize(cv::Mat patch);

void build_histogram(const std::vector<unsigned char>& textonz,
        cv::Mat mat, int i, int j);

cv::Mat lbp(cv::Mat image);
