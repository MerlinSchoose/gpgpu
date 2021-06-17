#include "nearest_neighbour.hh"

#include <limits>

cv::Mat nearest_neighbour(const cv::Mat &lbp, const cv::Mat &centroids)
{
    cv::Mat labels(lbp.rows, 1, CV_32S);

    for (int i = 0; i < lbp.rows; i++)
    {
        auto patch = lbp.row(i);

        int min_centroid_id = 0;
        double min_centroid_dist = std::numeric_limits<double>::max();

        for (int j = 0; j < centroids.rows; j++)
        {
            auto centroid = centroids.row(j);

            cv::Mat centroid_diff = (centroid - patch);
            double centroid_dist = centroid_diff.dot(centroid_diff);

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
