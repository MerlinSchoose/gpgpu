#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
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

int video_render_and_save(const std::string &output_path, const std::string &mode,
                          const std::string &filename, unsigned char *colors,
                          bool verbose)
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

        // BEGIN crappy shit
        cv::imshow( "Frame", rgb_frame);
        char c=(char)cv::waitKey(25);
        if(c==27)
            break;
        // END

        if (rgb_frame.empty())
            break;

        cv::cvtColor(rgb_frame, frame, cv::COLOR_BGR2GRAY);

        cv::Mat labels_mat;
        labels_mat = do_render(mode, frame, colors);

        // progress print
        if (verbose)
            std::cout << "\r" << "[" << i++ << " / " << nb_frames << "] frames rendered" << std::flush;

        video_output.write(labels_mat);

        if (mode == "GPU-OPTI")
            free(labels_mat.data);
    }
    if (verbose)
        std::cout << "\r" << nb_frames << " frames rendered" << std::endl;

    video_capture.release();
    video_output.release();

    if (verbose)
        std::cout << "Output saved in " << output_path << std::endl;

    return 0;
}