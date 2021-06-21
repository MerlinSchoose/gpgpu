#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <string>


cv::Mat pipepline_cpu(const cv::Mat &image, unsigned char *colors);
cv::Mat pipepline_gpu(const cv::Mat &image, unsigned char *colors);
cv::Mat pipepline_gpu_opti(const cv::Mat &image, unsigned char *colors);

int image_render_and_save(const std::string &output_path, const std::string &mode,
                          const std::string &filename, unsigned char *colors);
int video_render_and_save(const std::string &output_path, const std::string &mode,
                          const std::string &filename, unsigned char *colors,
                          bool verbose=true);