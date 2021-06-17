#include "utils.hh"

std::vector<unsigned char> mat_to_vect(cv::Mat image)
{
    cv::Mat flat = image.reshape(1, image.total() * image.channels());
    std::vector<unsigned char> vec = image.isContinuous()? flat : flat.clone();
    return vec;
}

cv::Mat vect_to_mat(std::vector<unsigned char> image)
{
    return cv::Mat();
}

unsigned char* mat_to_bytes(cv::Mat image)
{
    int image_size = image.total() * image.elemSize();
    unsigned char* image_uchar = new unsigned char[image_size];
    std::memcpy(image_uchar, image.data, image_size * sizeof(unsigned char));

    return image_uchar;
}

cv::Mat bytes_to_mat(unsigned char* image_uchar, int image_rows,
        int image_cols, int image_type)
{
    cv::Mat image(image_rows, image_cols, image_type, image_uchar,
            cv::Mat::AUTO_STEP);
    return image;
}
