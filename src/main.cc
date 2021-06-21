#include <CLI/CLI.hpp>

#include "pipeline.hh"

#include <string>
#include <opencv2/videoio.hpp>

cv::Mat do_render(const std::string &mode, const cv::Mat &image, unsigned char *colors)
{
    cv::Mat labels_mat;

    if (mode == "CPU")
        labels_mat = pipepline_cpu(image, colors);
    else if (mode == "GPU")
        labels_mat = pipepline_gpu(image, colors);
    else if (mode == "GPU-OPTI")
        labels_mat = pipepline_gpu_opti(image, colors);
    else
        exit(1);

    return labels_mat;
}

int image_render_and_save(const std::string &output_path, const std::string &mode, const std::string &filename, unsigned char *colors)
{
    const cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat labels_mat;
    labels_mat = do_render(mode, image, colors);

    cv::imwrite(output_path, labels_mat);

    if (mode == "GPU-OPTI")
        free(labels_mat.data);

    std::cout << "Output saved in " << output_path << std::endl;

    return 0;
}

int video_render_and_save(const std::string &output_path, const std::string &mode, const std::string &filename, unsigned char *colors)
{
    cv::VideoCapture video_capture(filename);
    if (!video_capture.isOpened())
    {
        std::cerr  << "Could not open the input video: " << filename << std::endl;
        return 1;
    }

    const auto nb_frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter video_output(output_path,
                                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                 video_capture.get(cv::CAP_PROP_FPS),
                                 cv::Size(video_capture.get(cv::CAP_PROP_FRAME_WIDTH),
                                          video_capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    auto i = 0;
    while (true)
    {
        cv::Mat rgb_frame, frame;
        video_capture >> rgb_frame;

        if (rgb_frame.empty())
            break;

        cv::cvtColor(rgb_frame, frame, cv::COLOR_BGR2GRAY);

        cv::Mat labels_mat;
        labels_mat = do_render(mode, frame, colors);

        // progress print
        std::cout << "\r" << "[" << i++ << " / " << nb_frames << "] frames rendered" << std::flush;

        video_output.write(labels_mat);

        if (mode == "GPU-OPTI")
            free(labels_mat.data);
    }
    std::cout << "\r" << nb_frames << " frames rendered" << std::endl;

    video_capture.release();
    video_output.release();

    std::cout << "Output saved in " << output_path << std::endl;

    return 0;
}

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
