//
//  Filters.h
//  Seminar
//
//  Created by Srđan Rašić on 4/21/12.
//

/*! Non-Maximum Suppression algorithm as defined in "Efficient Non-Maximum Suppression"
 *  by A. Neubeck and L. V. Gool (Algorithm 4):
 *  http://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00446.pdf
 */
__kernel void 
nms(__global uchar* image, __global uchar* maxima, unsigned int W, unsigned int H, int n)
{
    unsigned int u = get_global_id(0);
    unsigned int v = get_global_id(1);
    
    unsigned int i = n + u * (n + 1);
    unsigned int j = n + v * (n + 1);
    
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

/*! Simple 2D convolution
 */
__kernel void 
convolve2d(__global uchar* in, __global uchar* out, __constant uchar* conv_kernel, 
           int in_width, int width, int height, int kernel_size)
{     
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    const int y_top_left = y;
    const int x_top_left = x;
            
    unsigned char sum = 0;
    for (int yy = 0; yy < kernel_size; yy++) {
        const int kernel_row_index = yy * kernel_size;
        const int in_image_row_index = (y_top_left + yy) * in_width + x_top_left;
        
        for (int xx = 0; xx < kernel_size; xx++) {
            const int kernel_index = kernel_row_index + xx;
            const int in_image_index = in_image_row_index + xx;
            sum += conv_kernel[kernel_index] * in[in_image_index];
        }
    }
    
    const int out_image_index = y * width + x;
    out[out_image_index] = sum;

    
//	int j = get_global_id(0);
//    int i = get_global_id(1);
//    
//    __local uchar inBlock[5*5];
//    
//    int l_j = get_local_id(0);
//    int l_i = get_local_id(1);
//    
//    inBlock[l_i * W + l_j] = in[i * W + j];
//    barrier(CLK_LOCAL_MEM_FENCE);   
//    
//    int k_center_x = k_rows / 2;
//    int k_center_y = k_cols / 2;
//    
//    out[i * W + j] = 0;
//    
//    for (int m = 0; m < k_rows; m++) {
//        int mm = k_rows - 1 - m;
//        
//        for (int n = 0; n < k_cols; n++) {
//            int nn = k_cols - 1 - n;
//            
//            int ii = l_i + (m - k_center_y);
//            int jj = l_j + (n - k_center_x);
//            
//            //if (ii >= 0 && ii < H && jj >= 0 && jj < W)
//                out[i * W + j] += inBlock[ii * W + jj] * ker[mm * 5 + nn];
//        }
//    }
}
