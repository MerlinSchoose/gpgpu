#include <CLI/CLI.hpp>

#include "pipeline.hh"

/*
 * ./exe -m GPU 
 * ./exe -m CPU
 * ./exe -m GPU-OPTI
 */
int main(int argc, char** argv)
{
    const std::string output_dir = "../results/";
    const std::string output_name = "output";

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

    const auto input_extension = inputfilename.substr(inputfilename.find_last_of(".") + 1);

    const bool video_rendering = input_extension == "mp4";
    const std::string output_extension = video_rendering ? "mp4" : "png";
    const std::string output_path = output_dir + output_name + "." + output_extension;

    std::cout << "Rendering mode : " << (video_rendering ? "video" : "image") << std::endl;

    if (video_rendering)
        return video_render_and_save(output_path, mode, inputfilename, colors);

    return image_render_and_save(output_path, mode, inputfilename, colors);
}
