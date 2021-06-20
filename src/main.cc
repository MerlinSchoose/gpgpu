#include <CLI/CLI.hpp>
#include <serialize.hh>

#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "render.hh"
#include "utils.hh"
#include "render_gpu.hh"

/*
 * ./exe -m GPU 
 * ./exe -m CPU
 */
int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    std::string filename = "../results/output.png";
    std::string inputfilename = "../data/barcode-00-01.jpg";
    std::string mode = "GPU-OPTI";

    CLI::App app{"gpgpu"};
    app.add_option("-i", inputfilename, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU", "GPU-OPTI"}, "Either 'GPU', 'GPU-OPTI' or 'CPU'");

    CLI11_PARSE(app, argc, argv);

    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    cv::Mat histos_mat;
    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    cv::Mat labels_mat;

    unsigned char *histos_buffer = (unsigned char *)
            malloc(rows * cols * sizeof(unsigned char));

    if (mode == "CPU")
    {
        histos_mat = cpu_lbp(image);
    }
    else if (mode == "GPU")
    {
        gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
        histos_mat = bytes_to_mat(histos_buffer, cols, rows, image.type());
    }
    else if (mode == "GPU-OPTI")
    {
        unsigned char *histos_dev;
        size_t histos_pitch;
        gpu_lbp_opti(mat_to_bytes(image), image.cols, image.rows, &histos_dev, &histos_pitch);

        cv::Mat centers;
        deserializeMat(centers, "../results/.centroids");

        auto labels_mat_arr = render_gpu(image.cols, image.rows, histos_dev, histos_pitch,
                                         centers.ptr<float>(), colors);
        labels_mat = bytes_to_mat(labels_mat_arr, image.cols, image.rows, CV_8UC3);
    }
    else
    {
        std::cerr << "Invalid argument";
        return 1;
    }

    // Rendering
    if (mode != "GPU-OPTI")
        labels_mat = render(image, histos_mat);

    free(histos_buffer);

    // Save
    cv::imwrite(filename, labels_mat);
    std::cout << "Output saved in " << filename << "." << std::endl;

    return 0;
}
