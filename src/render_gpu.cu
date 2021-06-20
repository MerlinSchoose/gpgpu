#include "render_gpu.hh"
#include "utils_gpu.cuh"

#define K 16

__global__ void nearest_neighbour(unsigned char *histos, unsigned width, unsigned height, size_t histos_pitch,
                                  float *centroids, size_t centroids_pitch, unsigned *labels) {
    __shared__ unsigned min_index;
    __shared__ float min_dist;
    __shared__ float dist[K];

    if (threadIdx.x >= width || threadIdx.y >= height)
        return;

    if (threadIdx.x == 0) {
        min_index = 3;
        min_dist = INFINITY;
    }
    __syncthreads();

    unsigned char *histo = histos + blockIdx.x * histos_pitch;
    float val = (float) histo[threadIdx.x];

    for (unsigned i = 0; i < K; ++i) {
        float *centroid = centroids + i * centroids_pitch;
        float diff = val - centroid[threadIdx.x];
        diff *= diff;
        atomicAdd(&dist[i], diff);
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned i = 0; i < K; ++i) {
            if (min_dist > dist[i]) {
                min_index = i;
                min_dist = dist[i];
            }
        }
    }

    __syncthreads();
    labels[blockIdx.x] = min_index;
}

__global__ void colorize(unsigned char* image, unsigned width, unsigned height, size_t pitch,
                         const unsigned *labels, const unsigned char *colors) {
    __shared__ unsigned label;

    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned block_idx = (blockIdx.x + blockIdx.y * (width / blockDim.x));
        label = labels[block_idx];
    }
    __syncthreads();
    image[x * 3 + y * pitch] = colors[label * 3];
    image[x * 3 + y * pitch + 1] = colors[label * 3 + 1];
    image[x * 3 + y * pitch + 2] = colors[label * 3 + 2];
    __syncthreads();
}

unsigned char *render_gpu(unsigned width, unsigned height, unsigned char *histos,
                          size_t histos_pitch, float *centers, unsigned char *rand_colors) {
    cudaError_t error;

    size_t block_size = width / TILE_SIZE * height / TILE_SIZE;

    unsigned char *color_tab;
    error = cudaMalloc(&color_tab, K * sizeof(unsigned char) * 3);
    if (error)
        abortError("Error on cudaMalloc");

    error = cudaMemcpy(color_tab, rand_colors, K * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
    if (error)
        abortError("Error on cudaMemcpy2D");

    size_t pitch;
    float *centroids;

    error = cudaMallocPitch(&centroids, &pitch, HISTO_SIZE * sizeof(float), K);
    if (error)
        abortError("Error on cudaMallocPitch");

    error = cudaMemcpy2D(centroids, pitch, centers, HISTO_SIZE * sizeof(float), HISTO_SIZE * sizeof(float),
                         K, cudaMemcpyHostToDevice);
    if (error)
        abortError("Error on cudaMemcpy2D");

    unsigned *labels;
    error = cudaMalloc(&labels, block_size * sizeof(unsigned));
    if (error)
        abortError("Error on cudaMalloc");

    nearest_neighbour<<<block_size, 256>>>(histos, HISTO_SIZE, block_size, histos_pitch, centroids, pitch, labels);
    cudaDeviceSynchronize();

    if(cudaPeekAtLastError())
        abortError("Error on nearest neighbour kernel");

    unsigned char *image;
    size_t pitch_image;

    cudaMallocPitch(&image, &pitch_image, width * sizeof(unsigned char) * 3, height);
    if (error)
        abortError("Error on mallocPitch");

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(width / threads.x, height / threads.y);
    printf("x: %d, y: %d", blocks.x, blocks.y);

    colorize<<<blocks, threads>>>(image, width, height, pitch_image, labels, color_tab);
    cudaDeviceSynchronize();

    if(cudaPeekAtLastError())
        abortError("Error on colorize kernel");

    error = cudaFree(histos);
    if (error)
        abortError("Error on free");

    error = cudaFree(centroids);
    if (error)
        abortError("Error on free");


    error = cudaFree(color_tab);
    if (error)
        abortError("Error on free");

    error = cudaFree(labels);
    if (error)
        abortError("Error on free");

    unsigned char *imageHost = (unsigned char *) malloc(width * height * sizeof(uchar3));

    error = cudaMemcpy2D(imageHost, width * sizeof(uchar3), image, pitch_image,
                         width * sizeof(uchar3), height, cudaMemcpyDeviceToHost);
    if (error)
        abortError("Error on cudaMemcpy2D");

    error = cudaFree(image);
    if (error)
        abortError("Error on free");

    return imageHost;
}