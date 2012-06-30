//
//  Controller.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#ifndef OCLW_Controller_h
#define OCLW_Controller_h

#include "OpenCL.h"
#include "MemoryBuffer.h"
#include <vector>
#include <string>

namespace oclw {
    class Program;
    
    /*! OpenCL controller class.
     *  
     *  Does framework initialization and is responsible for creation
     *  and management of basic objects that one should use to perform
     *  tasks on GPU device (or any other OpenCL compatible device).
     *  
     *  Class is implemented as "singleton" so you can't instantiate it.
     *  Use shared() method to get a reference to the actual object.
     */
    class Controller {
    public:
        /*! Device information container.
         *  
         *  Holds basic information about found device.
         */
        class Info {
        public:
            char name[255];
            char vendor[255];
            unsigned int compute_units;
            unsigned long global_mem_size;
            unsigned long local_mem_size;
            unsigned long constant_mem_size;
            size_t max_work_group_size;
            size_t max_work_item_sizes[3];
            
            void print ();
        };
        
    private:
        cl_platform_id _platform;
        cl_device_id _device;
        cl_context _context;
        cl_command_queue _queue;
        
        /* We are keeping list of object we allocate so we
         * can follow philosphy "Who allocated should also deallocate."
         */
        std::vector<MemoryBuffer*> _memoryBuffers;
        std::vector<Program*> _programs;
        
    private:
        /* Initializes OpenCL framework.
         */
        Controller ();
        
    public:
        ~Controller ();
        
        /*! Gets a reference to the singleton.
         */
        static Controller* shared ();
        
        /*! Gets current device info.
         */
        Controller::Info getInfo ();
        
        /*! Creates new memory buffer object.
         */
        MemoryBuffer* createMemoryBuffer ();
        
        /*! Creates new allocated memory buffer object.
         *
         *  \param mode Access mode (READ, WRITE, READ_WRITE or HOST).
         *  \param size Size of memory buffer in bytes.
         *  \param data Pointer to the memory block that is going to be used as a shared memory if mode is set to HOST.
         *  If access mode is set to HOST than this needs to be != NULL, otherwise is must be NULL.
         */
        MemoryBuffer* createMemoryBuffer (MemoryBuffer::AccessMode mode, size_t size, void* data = NULL);
        
        /*! Creates new program object.
         */
        Program* createProgramObject ();
        
        cl_context context () const;
        cl_command_queue cmdQueue () const;
        cl_device_id device () const;
    };
}

#endif
