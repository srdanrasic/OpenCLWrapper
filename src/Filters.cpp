//
//  Filters.cpp
//  Seminar
//
//  Created by Srđan Rašić on 4/21/12.
//

#include <iostream>
#include "Filters.h"

#include <omp.h>

namespace seminar {
    
   
    static unsigned int min (unsigned int a, unsigned int b) {
        return (a < b) ? a : b;
    }
    
    static unsigned int max (unsigned int a, unsigned int b) {
        return (a > b) ? a : b;
    }
    
    void nsm (uint8_t* image, unsigned int W, unsigned int H, uint8_t* maxima, unsigned int n) {
        unsigned int i = n, j = n;
        
        #pragma omp parallel for
        for (unsigned int u = 0; u <= (W - 2*n)/(n+1); u++) {
            for (unsigned int v = 0; v <= (H - 2*n)/(n+1); v++) {
                i = n + u * (n + 1);
                j = n + v * (n + 1);
                
                unsigned int mi = i, mj = j;
                
                for (unsigned int i2 = i; i2 <= i + n; i2++)
                    for (unsigned int j2 = j; j2 <= j + n; j2++)
                        if (image[j2*W + i2] > image[mj*W + mi]) {
                            mi = i2;
                            mj = j2;
                        }
                
                for (unsigned int i2 = mi - n; i2 <= min (mi + n, W - 1); i2++)
                    for (unsigned int j2 = mj - n; j2 <= min (mj + n, H - 1); j2++)
                        if (image[j2*W + i2] > image[mj*W + mi])
                            goto failed;
                
                maxima[mj*W + mi] = 255;
                failed:;
            }
        }
    }
        
    void convolution2d (const uint8_t* in, uint8_t* out, const uint8_t* kernel, int in_width, int width, int height, int kernel_size) {
        int threads = omp_get_max_threads();

#pragma omp parallel for num_threads(threads)
        for (int y = 0; y < height; y++) {
            const int y_top_left = y;
            
            for (int x = 0; x < width; x++) {
                const int x_top_left = x;
                
                uint8_t sum = 0;
                for (int yy = 0; yy < kernel_size; yy++) {
                    const int kernel_row_index = yy * kernel_size;
                    const int in_image_row_index = (y_top_left + yy) * in_width + x_top_left;
                    
                    for (int xx = 0; xx < kernel_size; xx++) {
                        const int kernel_index = kernel_row_index + xx;
                        const int in_image_index = in_image_row_index + xx;
                        sum += kernel[kernel_index] * in[in_image_index];
                    }
                }
                
                const int out_image_index = y * width + x;
                out[out_image_index] = sum;
            }
        }
    }
}
