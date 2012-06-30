//
//  Kernel.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#ifndef OCLW_Kernel_h
#define OCLW_Kernel_h

#include "OpenCL.h"

namespace oclw {
    class Controller;
    class MemoryBuffer;
    
    /*! Encapsulates OpenCL kernel object.
     *  
     *  Kernel object can only be created by Program object of
     *  which kernel is a part.
     */
    class Kernel {
        friend class Controller;
        friend class Program;
        
    public:
        /*! Use this class when there is a need to define size of 1,
         *  2 or 3 dimensions. Use implemented static methods to get an object. For example:
         *  
         *  \code
         *  oclw::Kernel::NDRange size = oclw::Kernel::NDRange::range2D(32, 16);
         *  \endcode
         */
        class NDRange {
            unsigned int _dims;
            size_t* _sizes;
            
            NDRange (unsigned int dims);
        public:
            ~NDRange ();
            
            static NDRange range1D (size_t x);
            static NDRange range2D (size_t x, size_t y);
            static NDRange range3D (size_t x, size_t y, size_t z);
            
            /*! Returns a number of dimensions.
             */
            unsigned int dims () const;
            
            /*! Returns a pointer to an array of dims() elements.
             *  Each element contains size of one dimension.
             */
            size_t* sizes () const;
            
            /*! For each dimension size checks if its divisible
             *  by coresponding dimension size of passed NDRange object.
             *  
             *  \return true if divisible, false otherwise.
             */
            bool divisible (NDRange& range) const;
        };
        
    private:
        cl_kernel _id;
        
        Controller& _controller;
        
    private:
        /* Private constructor enforces integrity stability.
         * Can only be instantiated from Controller (friend).
         */
        Kernel (Controller& c, cl_kernel id);
        ~Kernel ();
        
        /* Clears Kernel from memory.
         */
        void release ();
        
    public:
        /*! Sets Kernel argument.
         *  
         *  \param index Index of argument as defined in kernel's source.
         *  \param size Size of argument in bytes. For example: if argument
         *  is of type int, pass sizeof(int).
         *  \param value Pointer to actual value. Will be copied at call.
         */
        void setArgument(uint32_t index, size_t size, const void* value);
        
        /*! Sets Kernel argument.
         *  
         *  \param index Index of argument as defined in kernel's source.
         *  \param value MemoryBuffer object to be passed as argument.
         */
        void setArgument(uint32_t index, MemoryBuffer& memoryBuffer);
        
        /*! Executes Kernel. OpenCL will automatically calculate local work group size.
         *  
         *  \param global_work_size Global work size.
         */
        void execute (NDRange global_work_size);
        
        /*! Executes Kernel with specified local work group size.
         *  
         *  \param global_work_size Global work size.
         *  \param local_work_size Local work size (number of work
         *  items per dimension in a work group).
         */
        void execute (NDRange global_work_size, NDRange local_work_size);
    };
}

#endif
