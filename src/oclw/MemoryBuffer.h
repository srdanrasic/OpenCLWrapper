//
//  MemoryBuffer.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#ifndef OCLW_MemoryBuffer_h
#define OCLW_MemoryBuffer_h

#include "OpenCL.h"

namespace oclw {
    class Controller;
       
    /*! Encapsulates OpenCL memory buffer object.
     *  
     *  Can only be created by Controller.
     *  Provides methods for simple allocation and management of data.
     */
    class MemoryBuffer {
        friend class Controller;
        
    public:
        /*! Specifies how memory will be used so OpenCL knows how to optimise its
         *  storage and usage.
         */
        enum AccessMode { 
            READ_WRITE = CL_MEM_READ_WRITE, /*!< If OpenCL program will write to and read from the buffer. */
            READ = CL_MEM_READ_ONLY,        /*!< If OpenCL program will only read from the buffer. */
            WRITE = CL_MEM_WRITE_ONLY,      /*!< If OpenCL program will only write to the buffer. */
            HOST = CL_MEM_USE_HOST_PTR      /*!< Use this to map host (system) memory to the OpenCL device.
                                                Memory will be shared between host and OpenCL device. Note: access to
                                                this type of memory buffer from OpenCL device is very slow unless device anyways
                                                uses host memory as its global memory (such as integrated GPUs) */
        };
        
    private:
        cl_mem _id;
        size_t _size;
        AccessMode _mode;
        
        Controller& _controller;
        
    private:
        /* Private constructor enforces integrity stability.
         * Can only be instantiated from Controller (friend).
         */
        MemoryBuffer (Controller& c);
        MemoryBuffer (Controller& c, AccessMode mode, size_t size, void* data = NULL);
        ~MemoryBuffer ();
        
        /** Release memory
         */
        void release ();
        
    public:
        /*! Allocates memory on the OpenCL device.
         *  
         *  \param mode Access mode. Note: if set to MemoryBuffer::HOST, data must be set (!= NULL).
         *  \param size Size of memory buffer in bytes.
         *  \param data Pointer to the memory block that is going to be used as a shared memory if mode is set to HOST.
         */
        void allocate (AccessMode mode, size_t size, void* data = NULL);
        
        /*! Copies data from host to OpenCL device.
         *  
         *  \param data Pointer to the data that needs to be copied.
         *  \param size Size of the data to copy in bytes.
         */
        void writeData (void* data, size_t size);
        
        /*! Copies data from OpenCL device to back to host.
         *  
         *  \param data Pointer to the the memory block where data will be copied.
         *  \param size Size of the data to copy in bytes.
         */
        void readData (void* data, size_t size);
        
        /*! Returns unique ID of MemoryBuffer object.
         */
        cl_mem id () const;
    };
}


#endif
