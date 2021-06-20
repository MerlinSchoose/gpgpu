#include "gpu_lbp.hh"
#include "utils_gpu.cuh"


__global__ void kernel_opti(unsigned char* image, int width, int height,
        size_t pitch, unsigned char *histos_buffer, size_t pitch_buffer) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned histo[HISTO_SIZE];

    auto current_index = threadIdx.x + threadIdx.y * blockDim.x;
    histo[current_index] = 0;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    tile[threadIdx.x][threadIdx.y] = image[x + y * pitch];
    __syncthreads();

    atomicInc(&(histo[get_texton(tile, threadIdx.y, threadIdx.x)]), 256);
    __syncthreads();

    unsigned char *lineptr = histos_buffer + (blockIdx.x + blockIdx.y
            * ((width) / blockDim.x)) * pitch_buffer;

    lineptr[current_index] = (unsigned char) histo[current_index];
}

void gpu_lbp_opti(unsigned char *image, int image_cols, int image_rows,
        unsigned char *histos_buffer) {
    cudaError_t error = cudaSuccess;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((image_cols + threads.x - 1) / threads.x,
            (image_rows + threads.y - 1) / threads.y);

    size_t pitch_image;
    size_t pitch_histos;
    unsigned char *dev_image;
    unsigned char *dev_histos;

    error = cudaMallocPitch(&dev_histos, &pitch_histos,
            HISTO_SIZE, blocks.x * blocks.y * sizeof(unsigned char));

    if (error)
        abortError("Fail buffer allocation");

    error = cudaMallocPitch(&dev_image, &pitch_image,
            image_cols * sizeof(unsigned char), image_rows);

    if (error)
        abortError("Fail buffer allocation");

    error = cudaMemcpy2D(dev_image, pitch_image, image,
            image_cols * sizeof(unsigned char),
            image_cols * sizeof(unsigned char), image_rows,
            cudaMemcpyHostToDevice);

    if (error)
        abortError("Fail buffer copy to device");

    kernel_opti<<<blocks, threads>>>(dev_image, image_cols, image_rows, pitch_image,
            dev_histos, pitch_histos);
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    error = cudaMemcpy2D(histos_buffer, HISTO_SIZE, dev_histos,
            pitch_histos, HISTO_SIZE,
            blocks.x * blocks.y * sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

    if (error)
        abortError("Unable to copy buffer back to memory");

    error = cudaFree(dev_image);
    error = cudaFree(dev_histos);

    if (error)
        abortError("Unable to free memory");
}
