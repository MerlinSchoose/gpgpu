#include "cpu_lbp.hh"
#include "train_kmeans.hh"
#include "nearest_neighbour.hh"

cv::Mat cpu_lbp(cv::Mat image)
{
    auto res = lbp(image);
    res.convertTo(res, CV_32F);

    std::array<cv::Scalar, 16> color_tab;
    cv::RNG rng(13);
    for (size_t i = 0; i < color_tab.size(); i++)
        color_tab[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

    auto [centers, labels] = kmeans(16, res);

    labels = nearest_neighbour(res, centers);

    cv::Mat labels_mat(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < image.cols / 16; i++)
    {
        for (int j = 0; j < image.rows / 16; j++)
        {
            cv::Rect patch(i * 16, j * 16, 16, 16);
            labels_mat(patch) = color_tab[labels.at<int>(0, i * image.rows / 16 + j)];
        }
    }

    return labels_mat;
}
