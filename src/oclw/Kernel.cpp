//
//  Kernel.cpp
//  OCLW
//
//  Created by Srđan Rašić on 4/03/12.
//

#include <iostream>

#include "Kernel.h"
#include "Exception.h"
#include "MemoryBuffer.h"
#include "Controller.h"

namespace oclw {

    Kernel::NDRange::NDRange (unsigned int dims) {
        _dims = dims;
        _sizes = new size_t[dims];
    }
    
    Kernel::NDRange::~NDRange () {
        delete[] _sizes;
    }
       
    Kernel::NDRange Kernel::NDRange::range1D (size_t x) {
        NDRange range(1);
        range._sizes[0] = x;
        return range;
    }
    
    Kernel::NDRange Kernel::NDRange::range2D (size_t x, size_t y) {
        NDRange range(2);
        range._sizes[0] = x;
        range._sizes[1] = y;
        return range;
    }
    
    Kernel::NDRange Kernel::NDRange::range3D (size_t x, size_t y, size_t z) {
        NDRange range(3);
        range._sizes[0] = x;
        range._sizes[1] = y;
        range._sizes[2] = z;
        return range;
    }
    
    unsigned int Kernel::NDRange::dims () const {
        return _dims;
    }
    
    size_t* Kernel::NDRange::sizes () const {
        return _sizes;
    }
    
    bool Kernel::NDRange::divisible (NDRange& range) const {
        if (range.dims() != dims())
            return false;
            
        for (int i = 0; i < dims(); i++)
            if (sizes()[i] % range.sizes()[i] != 0)
                return false;
                
        return true;
    }
    
    Kernel::Kernel (Controller& c, cl_kernel id) : _controller(c), _id(id) {
        
    }
    
    Kernel::~Kernel () {
        release();
    }

    void Kernel::release () {
        if (_id != 0)
            clReleaseKernel(_id);
    }

    void Kernel::setArgument(uint32_t index, size_t size, const void* value) {
        cl_int err = clSetKernelArg(_id, index, size, value);
        
        if (err != CL_SUCCESS)
            throw Exception("Could not set kernel argument.");
    }
    
    void Kernel::setArgument(uint32_t index, MemoryBuffer& memoryBuffer) {
        cl_mem id = memoryBuffer.id();
        setArgument(index, sizeof(cl_mem), &id);
    }
    
    void Kernel::execute (NDRange global_work_size) {
        cl_int err = clEnqueueNDRangeKernel(_controller.cmdQueue(), _id,
                        global_work_size.dims(), NULL, global_work_size.sizes(),
                        NULL, 0, NULL, NULL);
        
        clFinish(_controller.cmdQueue());
        
        if (err != CL_SUCCESS)
            throw Exception("Could not execute kernel.");
    }
    
    void Kernel::execute (NDRange global_work_size, NDRange local_work_size) {
        if (global_work_size.dims() != local_work_size.dims())
            throw Exception("Number of specified dimensions of global and local work range is not equal!");   
            
        if (!global_work_size.divisible(local_work_size))
            throw Exception("Global group size not divisible with local group size.");
    
        cl_int err = clEnqueueNDRangeKernel(_controller.cmdQueue(), _id,
                        global_work_size.dims(),    // working dimensions
                        NULL,                       // offset
                        global_work_size.sizes(),   // global work size
                        local_work_size.sizes(),    // local work size
                        0, NULL, NULL);
        
        clFinish(_controller.cmdQueue());
        
        switch (err) {
            case CL_SUCCESS:
                return;
                break;
            case CL_INVALID_KERNEL_ARGS:
                throw Exception("Invalid kernel arguments.");
            case CL_INVALID_WORK_DIMENSION:
                throw Exception("Invalid work dimension.");
            case CL_INVALID_WORK_GROUP_SIZE:
                throw Exception("Invalid work group size.");
            case CL_INVALID_WORK_ITEM_SIZE:
                throw Exception("Invalid local work group size.");
                
            default:
                throw Exception("Could not execute kernel.");
        }
    }
}
