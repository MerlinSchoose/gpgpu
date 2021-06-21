#include <vector>
#include <benchmark/benchmark.h>

#include "pipeline.hh"
#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "utils.hh"

const std::string inputfilename = "../data/barcode-00-01.jpg";
const std::string video_inputfilename = "../data/barcode-00.mp4";

void BM_Pipeline_cpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);


    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    for (auto _ : st)
        pipepline_cpu(image, colors);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

void BM_Pipeline_gpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);


    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    for (auto _ : st)
        pipepline_gpu(image, colors);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}
void BM_Pipeline_gpu_opti(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);


    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    cv::Mat res;
    for (auto _ : st)
        res = pipepline_gpu_opti(image, colors);

    free(res.data);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Pipeline_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Pipeline_gpu_opti)
->Unit(benchmark::kMillisecond)
->UseRealTime();


void BM_LBP_cpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    for (auto _ : st)
        cpu_lbp(image);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

void BM_LBP_gpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    unsigned char *histos_buffer = (unsigned char *)
        malloc(rows * cols * sizeof(unsigned char));

    for (auto _ : st) {
        gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);

    free(histos_buffer);
}
void BM_LBP_gpu_opti(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    for (auto _ : st)
    {
        unsigned char *histos_dev;
        size_t histos_pitch;
        gpu_lbp_opti(mat_to_bytes(image), image.cols, image.rows, &histos_dev, &histos_pitch);
        cudaFreeWrapper(histos_dev);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

BENCHMARK(BM_LBP_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_LBP_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_LBP_gpu_opti)
->Unit(benchmark::kMillisecond)
->UseRealTime();


void BM_Pipeline_gpu_opti_video(benchmark::State& st)
{
    std::array<unsigned char, 16 * 3> color_tab = { 0 };
    cv::RNG rng(13);
    for (auto & color : color_tab)
        color = rng.uniform(0, 255);

    unsigned char *colors = &color_tab[0];

    cv::Mat res;
    for (auto _ : st)
        res = video_render_and_save("/tmp/foo.mp4", "GPU-OPTI", video_inputfilename, colors, false);

    free(res.data);

    st.counters["fixme"] = benchmark::Counter(st.iterations(),
                                              benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Pipeline_gpu_opti_video)
->Unit(benchmark::kSecond)
->UseRealTime();

BENCHMARK_MAIN();
