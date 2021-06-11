#include "../include/lbp.hh"

int main() {
    cv::Mat image = cv::imread("../data/barcode-00-01.jpg",
            cv::IMREAD_GRAYSCALE);

    auto res = lbp(image);
    std::cout << res.row(0) << "\n";

    return 0;
}
