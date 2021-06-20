#include "nearest_neighbour.hh"
#include "serialize.hh"

#define TILE_SIZE 16

cv::Mat render(const cv::Mat &image, const cv::Mat &histos)
{
    cv::Mat histos_f32;
    histos.convertTo(histos_f32, CV_32F);

    std::array<cv::Scalar, 16> color_tab;
    cv::RNG rng(13);
    for (size_t i = 0; i < color_tab.size(); i++)
        color_tab[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                  rng.uniform(0, 255));

    cv::Mat centers;
    deserializeMat(centers, "../results/.centroids");

    auto labels = nearest_neighbour(histos_f32, centers);

    cv::Mat labels_mat(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < (image.rows) / TILE_SIZE; i++)
    {
        for (int j = 0; j < (image.cols) / TILE_SIZE; j++)
        {
            cv::Rect patch(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            auto index = labels.at<int>(0, i * (image.cols) / TILE_SIZE + j);
            labels_mat(patch) = color_tab[index];
        }
    }

    return labels_mat;
}
