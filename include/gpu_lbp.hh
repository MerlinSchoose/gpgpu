#pragma once

#include <vector>
#include <string>

void gpu_lbp(unsigned char *image, int image_cols, int image_rows,
        unsigned char *image_buffer);
void gpu_lbp_opti(unsigned char *image, int image_cols, int image_rows,
                  unsigned char **histos_dev, size_t *histos_pitch);

void cudaFreeWrapper(void *histo_dev);
