#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#define HISTO_SIZE 256

unsigned char get(const cv::Mat &patch, int i, int j);
unsigned char get_texton(const cv::Mat &patch, int i, int j);
std::vector<unsigned char> textonize(const cv::Mat &patch);

void build_histogram(const std::vector<unsigned char>& textonz,
        const cv::Mat &mat, int i, int j);

cv::Mat cpu_lbp(const cv::Mat &image);
