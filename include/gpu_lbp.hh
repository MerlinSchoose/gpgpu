#pragma once

#include <vector>
#include <string>

void gpu_lbp(unsigned char *image, int image_cols, int image_rows,
        unsigned char *image_buffer);
