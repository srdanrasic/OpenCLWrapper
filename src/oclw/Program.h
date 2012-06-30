//
//  Program.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#ifndef OCLW_Program_h
#define OCLW_Program_h

#include "OpenCL.h"
#include <vector>

namespace oclw {
    class Controller;
    class Kernel;

    /*! Encapsulates OpenCL program object.
     *
     *  Can only be created by Controller.
     *  Provides methods for simple compilation and kernel creation.
     */
    class Program {
        friend class Controller;
        
    private:
        cl_program _id;
        Controller& _controller;
        
        std::vector<Kernel*> _kernels;
        
    private:
        /* Private constructor enforces integrity stability.
         * Can only be instantiated from Controller (friend).
         */
        Program (Controller& c);
        ~Program ();
        
        /* Clears built program from memory.
         */
        void release ();
        
    public:
        /*! Compiles program from a source string.
         *  
         * \param source OpenCL source code.
         */
        void compileFromSourceString (const char* source);
        
        /*! Compiles program from a source file.
         *  
         * \param file_path Path of to the file that contains OpenCL source code. 
         */
        void compileFromSourceFile (const char* file_path);
        
        /*! Creates Kernel object defined in the source code.
         */
        Kernel* createKernel (const char* name);
    };
}

#endif
