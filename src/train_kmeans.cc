#include "../include/train_kmeans.hh"

std::pair<cv::Mat, cv::Mat> kmeans(size_t k, cv::Mat lbp)
{
    cv::Mat data;
    lbp.convertTo(data, CV_32F);

    cv::Mat centers;

    cv::Mat labels;

    cv::kmeans(data, k, labels,
               cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    return {centers, labels};
}
