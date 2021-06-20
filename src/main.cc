#include <CLI/CLI.hpp>

#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "render.hh"
#include "utils.hh"

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
    std::string mode = "GPU";

    CLI::App app{"gpgpu"};
    app.add_option("-i", inputfilename, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU", "GPU-OPTI"}, "Either 'GPU', 'GPU-OPTI' or 'CPU'");

    CLI11_PARSE(app, argc, argv);

    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    unsigned char *histos_buffer = (unsigned char *)
        malloc(rows * cols * sizeof(unsigned char));

    cv::Mat histos_mat;

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
        gpu_lbp_opti(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
        histos_mat = bytes_to_mat(histos_buffer, cols, rows, image.type());
    }
    else
    {
        std::cerr << "Invalid argument";
        return 1;
    }

    // Rendering
    auto labels_mat = render(image, histos_mat);

    // Save
    cv::imwrite(filename, labels_mat);
    std::cout << "Output saved in " << filename << "." << std::endl;

    free(histos_buffer);

    return 0;
}
