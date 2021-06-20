#include "cpu_lbp.hh"
#include "train_kmeans.hh"
#include "nearest_neighbour.hh"

unsigned char get(const cv::Mat &patch, int i, int j)
{
    return (i < 0 || j < 0 || i >= patch.cols || j >= patch.rows)
        ? 0 : patch.at<unsigned char>(i, j);
}

unsigned char get_texton(const cv::Mat &patch, int i, int j)
{
    unsigned char value = patch.at<unsigned char>(i, j);

    unsigned char texton = get(patch, i - 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j - 1) >= value;

    return texton;
}

std::vector<unsigned char> textonize(const cv::Mat &patch)
{
    std::vector<unsigned char> textonZ;

    for (int i = 0; i < patch.cols; ++i)
    {
        for (int j = 0; j < patch.rows; ++j)
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

cv::Mat cpu_lbp(const cv::Mat &image)
{
    auto height = ((image.rows / 16) + (image.rows % 16 != 0)) * 16;
    auto width = ((image.cols / 16) + (image.cols % 16 != 0)) * 16;

    cv::Mat image_reshape(height, width, image.type());
    cv::copyMakeBorder(image, image_reshape,
            0, height - image.rows,
            0, width - image.cols,
            cv::BORDER_CONSTANT);

    cv::Mat result;

    for (int i = 0; i < height; i += 16)
    {
        for (int j = 0; j < width; j += 16)
        {
            cv::Rect rect(j, i, 16, 16);
            auto patch = image_reshape(rect);

            auto textonz = textonize(patch);

            auto histo = build_histogram(textonz);
           // Potentially normalize histo.

            result.push_back(cv::Mat(histo).t());
        }
    }

    return result;
}
