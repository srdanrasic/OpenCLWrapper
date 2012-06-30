//
//  Filters.h
//  Seminar
//
//  Created by Srđan Rašić on 4/21/12.
//

#ifndef Seminar_Filters_h
#define Seminar_Filters_h

#include <stdint.h>

namespace seminar {
    
    /*! Non-Maxima Suppresion algorithm implemented as in libviso2 (matcher.cpp) but
     *  does NMS (maxumim only) for one image. Function in libviso2 simultaneously 
     *  calculates max and min in two images. Algorithm is defined in "Efficient
     *  Non-Maximum Suppression" by A. Neubeck and L. V. Gool (Algorithm 4):
     *  http://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00446.pdf
     */
    void nsm (uint8_t* image, unsigned int width, unsigned int height, uint8_t* maxima, unsigned int nms_n);
    
    /*! Simple 2D convolution algorithm
     */
    void convolution2d (const uint8_t* in, uint8_t* out, const uint8_t* kernel, int in_width, int width, int height, int kernel_size);
}

#endif
