#include <CLI/CLI.hpp>

#include "pipeline.hh"


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

    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);


    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    if (mode == "CPU")
    {
        auto labels_mat = pipepline_cpu(image, colors);
        cv::imwrite(filename, labels_mat);
    }
    else if (mode == "GPU")
    {
        auto labels_mat = pipepline_gpu(image, colors);
        cv::imwrite(filename, labels_mat);
    }
    else if (mode == "GPU-OPTI")
    {
        auto labels_mat = pipepline_gpu_opti(image, colors);
        cv::imwrite(filename, labels_mat);
        free(labels_mat.data);
    }
    else
    {
        std::cerr << "Invalid argument";
        return 1;
    }
    std::cout << "Output saved in " << filename << "." << std::endl;

    return 0;
}
