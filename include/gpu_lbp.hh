#pragma once

#include <vector>
#include <string>

#define HISTO_SIZE 256
#define THREADS_SIZE 32
#define TILE_SIZE 16

void gpu_lbp(unsigned char *image, int image_cols, int image_rows,
        unsigned char *image_buffer);
