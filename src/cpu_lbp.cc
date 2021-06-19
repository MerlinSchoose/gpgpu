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
            textonZ.push_back(get_texton(patch, i, j));
        }
    }

    return textonZ;
}

std::array<unsigned char, HISTO_SIZE> build_histogram(
        const std::vector<unsigned char>& textonz)
{
    std::array<unsigned char, HISTO_SIZE> histo{ 0 };

    for (size_t i = 0; i < textonz.size(); ++i)
        histo[textonz[i]]++;

    return histo;
}

cv::Mat cpu_lbp(cv::Mat image)
{
    auto height = ((image.rows / 16) + (image.rows % 16 != 0)) * 16;
    auto width = ((image.cols / 16) + (image.cols % 16 != 0)) * 16;

    cv::Mat image_reshape(height, width, image.type());
    cv::copyMakeBorder(image, image_reshape,
            0, height - image.rows,
            0, width - image.cols,
            cv::BORDER_CONSTANT);

    cv::Mat result;

    for (int i = 0; i < width; i += 16)
    {
        for (int j = 0; j < height; j += 16)
        {
            cv::Rect rect(i, j, 16, 16);
            auto patch = image_reshape(rect);

            auto textonz = textonize(patch);

            auto histo = build_histogram(textonz);
            // Potentially normalize histo.

            result.push_back(cv::Mat(histo).t());
        }
    }

    return labels_mat;
}
