#include <opencv4/opencv2/opencv.hpp>
#include <serialize.hh>

#include "render.hh"
#include "render_gpu.hh"
#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "utils.hh"

cv::Mat pipepline_cpu(const cv::Mat &image, unsigned char *colors)
{
    cv::Mat histos_mat = cpu_lbp(image);
    cv::Mat res_mat = render(image, histos_mat, colors);
    return res_mat;
}

cv::Mat pipepline_gpu(const cv::Mat &image, unsigned char *colors)
{

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    unsigned char *histos_buffer = (unsigned char *)
        malloc(rows * cols * sizeof(unsigned char));


    gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
    cv::Mat histos_mat = bytes_to_mat(histos_buffer, cols, rows, image.type());
    cv::Mat res_mat = render(image, histos_mat, colors);

    free(histos_buffer);

    return res_mat;
}

cv::Mat pipepline_gpu_opti(const cv::Mat &image, unsigned char *colors)
{
    unsigned char *histos_dev;
    size_t histos_pitch;
    gpu_lbp_opti(mat_to_bytes(image), image.cols, image.rows, &histos_dev, &histos_pitch);

    cv::Mat centers;
    deserializeMat(centers, "../results/.centroids");

    auto labels_mat_arr = render_gpu(image.cols, image.rows, histos_dev, histos_pitch,
              centers.ptr<float>(), colors);

    return bytes_to_mat(labels_mat_arr, image.cols, image.rows, CV_8UC3);
}
