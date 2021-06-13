#include "../include/train_kmeans.hh"

std::pair<cv::Mat, cv::Mat> kmeans(size_t k, cv::Mat lbp)
{
    cv::Mat centers;

    cv::Mat labels;

    cv::kmeans(lbp, k, labels,
               cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    return {centers, labels};
}
