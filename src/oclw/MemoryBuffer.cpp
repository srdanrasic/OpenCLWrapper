//
//  MemoryBuffer.cpp
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#include <iostream>

#include "OpenCL.h"
#include "Exception.h"
#include "MemoryBuffer.h"
#include "Controller.h"

namespace oclw {
    MemoryBuffer::MemoryBuffer (Controller& c) : _controller(c), _id(0) {
    }
    
    MemoryBuffer::MemoryBuffer (Controller& c, AccessMode mode, size_t size, void* data) : _controller(c), _id(0) {
        allocate(mode, size, data);
    }
    
    MemoryBuffer::~MemoryBuffer () {
        release();
    }

    void MemoryBuffer::release () {
        if (_id != 0)
            clReleaseMemObject(_id);
    }
        
    void MemoryBuffer::allocate (AccessMode mode, size_t size, void* data) {
        /* If already allocated, deallocate */
        release();
        
        int err;
        _id = clCreateBuffer(_controller.context(), mode, size, data, &err);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not allocate memory buffer.");
    }

    void MemoryBuffer::writeData (void* data, size_t size) {
        cl_int err = clEnqueueWriteBuffer(_controller.cmdQueue(), _id, CL_TRUE, 0, size, data, 0, NULL, NULL);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not write data to memory buffer. Not allocated?");
    }

    void MemoryBuffer::readData (void* data, size_t size) {
        cl_int err = clEnqueueReadBuffer(_controller.cmdQueue(), _id, CL_TRUE, 0, size, data, 0, NULL, NULL);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not read data from memory buffer. Not allocated?");
    }
    
    cl_mem MemoryBuffer::id() const {
        return _id;
    }
}
