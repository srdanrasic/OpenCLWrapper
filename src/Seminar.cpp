//
//  Seminar.cpp
//  Seminar
//
//  Created by Srđan Rašić on 4/02/12.
//

#include <iostream>
#include <sys/time.h>
#include <stdlib.h>

#include <png++/png.hpp>

#include "oclw/Controller.h"
#include "oclw/MemoryBuffer.h"
#include "oclw/Program.h"
#include "oclw/Kernel.h"
#include "oclw/Exception.h"

#include "Filters.h"

/*! Simple timer class. Use tick() and tock() 
 *  methods to measure time.
 */
class Clock {
private:
    timeval _start_time;
    
public:
    /*! Starts clock.
     */
    void tick() {
        gettimeofday(&_start_time, NULL);
    }
    
    /*! Gets elapsed time since start or since last call of itself.
     *  \param ret Elapsed time since last call to tick().
     */
    void tock(double& ret) {
        timeval end_time;
        long seconds, useconds;
        
        gettimeofday(&end_time, NULL);
        seconds = end_time.tv_sec - _start_time.tv_sec;
        useconds = end_time.tv_usec - _start_time.tv_usec;
        
        gettimeofday(&_start_time, NULL);
        
        ret = seconds * 1000.0 + useconds / 1000.0;
    }
};


/* Converts uint8 array to png image object */
png::image<png::gray_pixel> uint8_to_png (uint8_t* image, unsigned int width, unsigned int height);


/* Entry point */
int main (int argc, const char * argv[]) {
    
#pragma mark Initialization
    Clock clock;
    double cpu_time, gpu_time;
    
    oclw::Controller* gpu_controller;
    oclw::Program* gpu_program;
    oclw::Kernel* nms_task_kernel, * cnv_task_kernel;
    
    /* Check args */
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " png_image_path nms_block_size" << std::endl;
        return -1;
    }
    
    /* Init GPU framework */
    try {
        /* On first call to shered(), OpenCL initialization is performed */
        gpu_controller = oclw::Controller::shared();
        gpu_controller->getInfo().print();
        gpu_program = gpu_controller->createProgramObject();
        gpu_program->compileFromSourceFile("src/cl_program.cl");
        nms_task_kernel = gpu_program->createKernel("nms");
        cnv_task_kernel = gpu_program->createKernel("convolve2d");
    } catch (oclw::Exception e) {
        std::cout << "OpenCL initialization error: " << e.what() << std::endl;
        return 0;
    }
    
#pragma mark Data loading
    /* Kernel used for convolution 2D */
    int kernel_size = 5; /* Width and height */
    uint8_t kernel[25] __attribute__ ((aligned (16))) = { 
        -1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1,
        -1,  1,  8,  1, -1,
        -1,  1,  1,  1, -1,
        -1, -1, -1, -1, -1 };
    
    /* Load image into uint8 array (grayscale) */
    png::image<png::gray_pixel> test_img_png;
    try {
        test_img_png = png::image<png::gray_pixel>(argv[1]);
    } catch (png::std_error e) {
        std::cout << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    unsigned int height = (unsigned int)test_img_png.get_height();
    unsigned int width = (unsigned int)test_img_png.get_width();
    
    unsigned int local_work_size_x = 15;
    unsigned int local_work_size_y = 15;
    
    /* Increase image size to be divisible by local_work_size */
    height += (height % local_work_size_y != 0) ? local_work_size_y - height % local_work_size_y : 0;
    width += (width % local_work_size_x != 0) ? local_work_size_x - width % local_work_size_x : 0;
    height += kernel_size - 1;
    width += kernel_size - 1;
    
    uint8_t* test_img __attribute__ ((aligned (16))) = new uint8_t[width * height];
    uint8_t* out_img __attribute__ ((aligned (16))) = new uint8_t[width * height];
    
    /* Convert image to uint8 array and init out_img array to zeroes */
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++) {
            try {
                test_img[i*width + j] = test_img_png.get_pixel(j, i);
            } catch (std::out_of_range e) {
                test_img[i*width + j] = 0;
            }
            out_img[i*width + j] = 0;
        }
    
    /* Create memory objects on GPU and transfer test_img, out_img and kernel to them */
    clock.tick();
    
    oclw::MemoryBuffer* test_img_gpu;
    oclw::MemoryBuffer* out_img_gpu;
    oclw::MemoryBuffer* kernel_gpu;
    
    try {
        test_img_gpu = gpu_controller->createMemoryBuffer(oclw::MemoryBuffer::READ, sizeof(uint8_t)*width*height);
        test_img_gpu->writeData(test_img, sizeof(uint8_t)*width*height);
        
        out_img_gpu = gpu_controller->createMemoryBuffer(oclw::MemoryBuffer::READ_WRITE, sizeof(uint8_t)*width*height);
        out_img_gpu->writeData(out_img, sizeof(uint8_t)*width*height);
        
        kernel_gpu = gpu_controller->createMemoryBuffer(oclw::MemoryBuffer::READ, sizeof(uint8_t)*kernel_size*kernel_size);
        kernel_gpu->writeData(kernel, sizeof(uint8_t)*kernel_size*kernel_size);
    } catch (oclw::Exception e) {
        std::cout << "GPU memory initialization error: " << e.what() << std::endl;
        return 0;
    }
    
    clock.tock(gpu_time);
    std::cout << std::endl << "Data transfer to the OpenCL device memory completed in " << gpu_time << " ms" << std::endl;
    
#pragma mark Testing: Non-Maximum Suppression
    int n;  /* NMS block size */
    sscanf(argv[2], "%d", &n);
    
    std::cout << "\nStarting Non-Maximum Suppression algorithm test (n = " << n << ")" << std::endl;
    std::cout << "Performing operations on the CPU and on the OpenCL device" << std::endl;
    
    /* Perform calculation on CPU */
    clock.tick();
    seminar::nsm(test_img, width, height, out_img, n);
    clock.tock(cpu_time);
    
    uint8_to_png(out_img, width, height).write("resources/test_image_nms_cpu.png");
    
    /* Perform calculation on GPU */
    nms_task_kernel->setArgument(0, *test_img_gpu);
    nms_task_kernel->setArgument(1, *out_img_gpu);
    nms_task_kernel->setArgument(2, sizeof(int), &width);
    nms_task_kernel->setArgument(3, sizeof(int), &height);
    nms_task_kernel->setArgument(4, sizeof(int), &n);
    
    clock.tick();
    nms_task_kernel->execute(oclw::Kernel::NDRange::range2D((width - 2*n)/(n+1)+1, (height - 2*n)/(n+1)+1));
   
    clock.tock(gpu_time);
    
    out_img_gpu->readData(out_img, width*height);
    uint8_to_png(out_img, width, height).write("resources/test_image_nms_gpu.png");
    
    /* Print results */
    std::cout << "CPU running time: " << cpu_time << " ms" << std::endl;
    std::cout << "OpenCL device running time: " << gpu_time << " ms" << std::endl;
    
    
#pragma mark Testing: Convolution 2D   
    std::cout << "\nStarting Convolution 2D algorithm test" << std::endl;
    std::cout << "Performing operations on the CPU and on the OpenCL device" << std::endl;
    
    /* Clear out_img */
    memset(out_img, 0, width*height);
    
    const int out_width = width - kernel_size + 1;
    const int out_height = height - kernel_size + 1;
    
    /* Perform calculation on CPU (simple version) */
    clock.tick();
    seminar::convolution2d(test_img, out_img, kernel, width, out_width, out_height, kernel_size);
    clock.tock(cpu_time);
    
    uint8_to_png(out_img, out_width, out_height).write("resources/test_image_blob_cpu.png");
    
    /* Perform calculation on GPU (simple version) */
    cnv_task_kernel->setArgument(0, *test_img_gpu);
    cnv_task_kernel->setArgument(1, *out_img_gpu);
    cnv_task_kernel->setArgument(2, *kernel_gpu);
    cnv_task_kernel->setArgument(3, sizeof(int), &width);
    cnv_task_kernel->setArgument(4, sizeof(int), &out_width);
    cnv_task_kernel->setArgument(5, sizeof(int), &out_height);
    cnv_task_kernel->setArgument(6, sizeof(int), &kernel_size);
    
    try {
        clock.tick();
        cnv_task_kernel->execute(oclw::Kernel::NDRange::range2D(out_width, out_height),
                                 oclw::Kernel::NDRange::range2D(local_work_size_x, local_work_size_y));
        clock.tock(gpu_time);
    } catch (oclw::Exception e) {
        std::cout << "Executing kernel error: " << e.what() << std::endl;
        return 0;
    }
    
    out_img_gpu->readData(out_img, width*height);
    uint8_to_png(out_img, out_width, out_height).write("resources/test_image_blob_gpu.png");
    
    /* Print results */
    std::cout << "CPU running time: " << cpu_time << " ms" << std::endl;
    std::cout << "OpenCL device running time: " << gpu_time << " ms" << std::endl;

    
#pragma mark Finalize    
    /* Delete allocated objects */
    delete[] test_img;
    delete[] out_img;
    
    return 0;
}


png::image<png::gray_pixel> uint8_to_png (uint8_t* image, unsigned int width, unsigned int height) {
    png::image<png::gray_pixel> png_image(width, height);
    
    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            png_image.set_pixel(j, i, image[i*width + j]);
    
    return png_image;
}
