#include <CLI/CLI.hpp>
#include <serialize.hh>

#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "render.hh"
#include "utils.hh"
#include "render_gpu.hh"

#include <string>


cv::Mat do_cpu(cv::Mat image, unsigned char *colors)
{
    cv::Mat histos_mat = cpu_lbp(image);
    return render(image, histos_mat, colors);
}

cv::Mat do_gpu(cv::Mat image, unsigned char *colors)
{

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    unsigned char *histos_buffer = (unsigned char *)
        malloc(rows * cols * sizeof(unsigned char));


    gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
    cv::Mat histos_mat = bytes_to_mat(histos_buffer, cols, rows, image.type());
    cv::Mat labels_mat = render(image, histos_mat, colors);
    free(histos_buffer);

    return labels_mat;
}

cv::Mat do_gpu_opti(cv::Mat image, unsigned char *colors)
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

cv::Mat do_render(const std::string &mode, const cv::Mat &image, unsigned char *colors)
{
    cv::Mat labels_mat;

    if (mode == "CPU")
        labels_mat = do_cpu(image, colors);
    else if (mode == "GPU")
        labels_mat = do_gpu(image, colors);
    else if (mode == "GPU-OPTI")
        labels_mat = do_gpu_opti(image, colors);
    else
        exit(1);

    return labels_mat;
}

/*
 * ./exe -m GPU 
 * ./exe -m CPU
 * ./exe -m GPU-OPTI
 */
int main(int argc, char** argv)
{
    std::string filename = "../results/output.png";
    std::string inputfilename = "../data/barcode-00-01.jpg";
    std::string mode = "GPU-OPTI";

    CLI::App app{"gpgpu"};
    app.add_option("-i", inputfilename, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU", "GPU-OPTI"}, "Either 'GPU', 'GPU-OPTI' or 'CPU'");

    CLI11_PARSE(app, argc, argv);

    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    cv::Mat labels_mat;
    labels_mat = do_render(mode, image, colors);

    // Save
    cv::imwrite(filename, labels_mat);
    std::cout << "Output saved in " << filename << "." << std::endl;

    return 0;
}
