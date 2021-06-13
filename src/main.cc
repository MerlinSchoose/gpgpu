#include "../include/lbp.hh"
#include "../include/train_kmeans.hh"

int main() {
    cv::Mat image = cv::imread("../data/barcode-00-02.jpg",
            cv::IMREAD_GRAYSCALE);

    auto res = lbp(image);

    auto [centers, labels] = kmeans(16, res);

    cv::Scalar color_tab[] = {
            cv::Scalar(0, 0, 255),
            cv::Scalar(0,255,0),
            cv::Scalar(255,100,100),
            cv::Scalar(255,0,255),
            cv::Scalar(0,255,255),
            cv::Scalar(255,0,0),
            cv::Scalar(255,255,0),
            cv::Scalar(255,255,255)
    };

    cv::Mat labels_mat(image.rows, image.cols, CV_8UC3);

    for (int i = 0; i < image.cols / 16; i++)
    {
        for (int j = 0; j < image.rows / 16; j++)
        {
            cv::Rect patch(i * 16, j * 16, 16, 16);
            labels_mat(patch) = color_tab[labels.at<int>(0, i * image.rows / 16 + j)];
        }
    }

    cv::imwrite("labels mat 2.png", labels_mat);
    cv::waitKey(0);

    return 0;
}
