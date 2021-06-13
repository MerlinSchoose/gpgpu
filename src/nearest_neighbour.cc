#include "../include/nearest_neighbour.hh"

#include <limits>

cv::Mat nearest_neighbour(const cv::Mat &lbp, const cv::Mat &centroids)
{
    cv::Mat labels(lbp.rows, 1, CV_8U);

    for (int i = 0; i < lbp.rows; i++)
    {
        auto patch = lbp.row(i);

        unsigned char min_centroid_id = 0;
        float min_centroid_dist = std::numeric_limits<float>::max();

        for (unsigned char j = 0; j < centroids.rows; j++)
        {
            auto centroid = centroids.row(j);

            cv::Mat centroid_diff = (centroid - patch);
            float centroid_dist = centroid_diff.dot(centroid_diff);

            if (centroid_dist < min_centroid_dist)
            {
                min_centroid_id = j;
                min_centroid_dist = centroid_dist;
            }
        }

        labels.push_back(min_centroid_id);
    }

    return labels.t();
}
