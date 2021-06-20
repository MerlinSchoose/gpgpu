#include "utils_gpu.cuh"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::cerr << msg << " (" << fname << ", line: " << line << ")\n";
    std::cerr << "Error " << cudaGetErrorName(err) << ": "
        << cudaGetErrorString(err) << "\n";
    std::exit(1);
}

__device__ unsigned char get(unsigned char patch[TILE_SIZE][TILE_SIZE], int i,
        int j)
{
    return (i < 0 || j < 0 || i >= TILE_SIZE || j >= TILE_SIZE)
        ? 0 : patch[j][i];
}

__device__ unsigned char get_texton(unsigned char patch[TILE_SIZE][TILE_SIZE],
        int i, int j)
{
    unsigned char value = patch[j][i];

    unsigned char texton = get(patch, i - 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j - 1) >= value;

    return texton;
}



/*cv::Mat nearest_neighbour(const cv::Mat &lbp, const cv::Mat &centroids)
{
    cv::Mat labels;

    for (int i = 0; i < lbp.rows; i++)
    {
        auto patch = lbp.row(i);

        int min_centroid_id = 0;
        double min_centroid_dist = std::numeric_limits<double>::max();

        for (int j = 0; j < centroids.rows; j++)
        {
            auto centroid = centroids.row(j);

            cv::Mat centroid_diff = (centroid - patch);
            double centroid_dist = centroid_diff.dot(centroid_diff);

            if (centroid_dist < min_centroid_dist)
            {
                min_centroid_id = j;
                min_centroid_dist = centroid_dist;
            }
        }

        labels.push_back(min_centroid_id);
    }

    return labels.t();
}*/
