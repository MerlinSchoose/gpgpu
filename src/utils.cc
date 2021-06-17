#include "utils.hh"

std::vector<unsigned char> mat_to_vect(cv::Mat image)
{
    cv::Mat flat = image.reshape(1, image.total()*image.channels());
    std::vector<unsigned char> vec = image.isContinuous()? flat : flat.clone();
    return vec;
}

cv::Mat vect_to_mat(std::vector<unsigned char> image)
{
    return cv::Mat();
}
