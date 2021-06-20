#include <vector>
#include <string>

#include "gpu_lbp.hh"
#include "utils.hh"
#include "serialize.hh"
#include "train_kmeans.hh"

#define K 16

/*
 * ./exe -m GPU 
 * ./exe -m CPU
 */
int main(int argc, char** argv)
{
    std::vector<std::string> inputfilenames;

    int i = 1;
    for (; i < argc; i++)
    {
        inputfilenames.push_back(argv[i]);
    }
    if (i == 1)
        inputfilenames.push_back("../data/barcode-00-01.jpg");

    cv::Mat histos_mat;

    for (auto filename : inputfilenames)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
            * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
        size_t cols = HISTO_SIZE;

        unsigned char *histos_buffer = (unsigned char *)
            malloc(rows * cols * sizeof(unsigned char));


        gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);
        histos_mat.push_back(bytes_to_mat(histos_buffer, cols, rows, image.type()));

        free(histos_buffer);
    }


    cv::Mat histos_f32;
    histos_mat.convertTo(histos_f32, CV_32F);

    auto [centroids, labels] = kmeans(K, histos_f32);
    
    serializeMat("../results/.centroids", centroids);

    return 0;
}
