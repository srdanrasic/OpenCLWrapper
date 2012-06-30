//
//  Controller.cpp
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#include <iostream>
#include <assert.h>

#include "Controller.h"
#include "MemoryBuffer.h"
#include "Program.h"
#include "Exception.h"

namespace oclw {

    void Controller::Info::print () {
        std::cout << "OpenCL device:\n" << vendor << " " << name << std::endl;
        std::cout << "Compute units: " << compute_units << std::endl;
        std::cout << "Global memory size: " << global_mem_size / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Local memory size: " << local_mem_size / 1024 << " KB" << std::endl;
        std::cout << "Constant memory size: " << constant_mem_size / 1024 << " KB" << std::endl;
        std::cout << "Max work group size: " << max_work_group_size << " work-items" << std::endl;
        std::cout << "Max number of work items per dimension: [" << max_work_item_sizes[0] << ", "
                                              << max_work_item_sizes[1] << ", "
                                              << max_work_item_sizes[2] << "]" << std::endl;
    }
    
    static Controller* _instance = NULL;
    
    Controller* Controller::shared () {
        if (_instance == NULL)
            _instance = new Controller ();
        
        return _instance;
    }
    
    Controller::Controller () {
        cl_int err;
        err = clGetPlatformIDs(1, &_platform, NULL);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not get platform information.");
        
        err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device , NULL);
        
        /* If GPU unavailable, fallback to CPU
         */
        if (err == CL_DEVICE_NOT_FOUND)
            err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_CPU, 1, &_device , NULL);

        if (err != CL_SUCCESS)
            throw Exception("Could not find any comaptible computation device.");
        
        _context = clCreateContext(0, 1, &_device, NULL, NULL, &err);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not create OpenCL context.");
        
        _queue = clCreateCommandQueue(_context, _device, 0, &err);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not create OpenCL command queue.");
    }
    
    Controller::~Controller () {
        /* Delete allocated memory buffer objects
         */
        for (int i = 0; i < _memoryBuffers.size(); i++)
            delete _memoryBuffers[i];
        
        /* Delete allocated program objects
         */
        for (int i = 0; i < _programs.size(); i++)
            delete _programs[i];
        
        /* Teardown Other stuff
         */
        clReleaseCommandQueue(_queue);
        clReleaseContext(_context);
    }
    
    Controller::Info Controller::getInfo () {
        Controller::Info info;
        cl_int err = 0;
        
        err |= clGetDeviceInfo(_device, CL_DEVICE_NAME, 
                sizeof(info.name), info.name, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 
                sizeof(info.vendor), info.vendor, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_MAX_COMPUTE_UNITS, 
                sizeof(info.compute_units), &info.compute_units, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                sizeof(info.global_mem_size), &info.global_mem_size, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, 
                sizeof(info.local_mem_size), &info.local_mem_size, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 
                sizeof(info.constant_mem_size), &info.constant_mem_size, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                sizeof(info.max_work_group_size), &info.max_work_group_size, NULL);
        err |= clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
                sizeof(info.max_work_item_sizes), &info.max_work_item_sizes, NULL);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not read device info.");
            
        return info;
    }
    
    MemoryBuffer* Controller::createMemoryBuffer () {
        MemoryBuffer* memoryBuffer = new MemoryBuffer(*this);
        _memoryBuffers.push_back(memoryBuffer);
        return memoryBuffer;
    }
    
    MemoryBuffer* Controller::createMemoryBuffer (MemoryBuffer::AccessMode mode, size_t size, void* data) {
        MemoryBuffer* memoryBuffer = new MemoryBuffer(*this, mode, size, data);
        _memoryBuffers.push_back(memoryBuffer);
        return memoryBuffer;
    }
    
    Program* Controller::createProgramObject () {
        Program* program = new Program(*this);
        _programs.push_back(program);
        return program;
    }
    
    cl_context Controller::context () const {
        return _context;
    }
    
    cl_command_queue Controller::cmdQueue () const {
        return _queue;
    }
    
    cl_device_id Controller::device () const {
        return _device;
    }
}
