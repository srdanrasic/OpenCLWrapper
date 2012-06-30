//
//  Program.cpp
//  OCLW
//
//  Created by Srđan Rašić on 4/03/12.
//

#include <iostream>
#include <fstream>

#include <string.h>

#include "Program.h"
#include "Controller.h"
#include "Exception.h"
#include "Kernel.h"

namespace oclw {
    Program::Program (Controller& c) : _controller (c) {
        _id = 0;
    }
    
    Program::~Program () {
        release();
    }

    void Program::release () {
        if (_id != 0) {            
            /* Delete allocated memory buffer objects
             */
            for (int i = 0; i < _kernels.size(); i++)
                delete _kernels[i];
            
            /* Delete program
             */
            clReleaseProgram(_id);
        }
    }
    
    void Program::compileFromSourceString (const char* source) {
        /* If already allocated, deallocate */
        release();
        
        cl_int err;
        _id = clCreateProgramWithSource(_controller.context(), 1, &source, NULL, &err);
        
        err |= clBuildProgram(_id, 0, NULL, NULL, NULL, NULL);
        
        if (err != CL_SUCCESS) {
            char msg[2048];
            char err_msg[] = "Error while compiling program:\n";
            
            strcpy (msg, err_msg);
            clGetProgramBuildInfo(_id, _controller.device(), CL_PROGRAM_BUILD_LOG, sizeof(msg), msg + sizeof(err_msg) - 1, NULL);
            throw Exception(msg);
        }
    }

    void Program::compileFromSourceFile (const char* file_path) {
        char* source;
        
        std::ifstream file (file_path, std::ios::in | std::ios::binary | std::ios::ate);
        
        if (file.is_open()) {
            std::ifstream::pos_type size = file.tellg();
            source = new char [(size_t)size];
            file.seekg(0, std::ios::beg);
            file.read(source, size);
            file.close();
        } else {
            throw Exception("Unable to open source file.");
        }
        
        compileFromSourceString(source);
    }

    Kernel* Program::createKernel (const char* name) {
        cl_kernel kernel_id;
        cl_int err;
        
        kernel_id = clCreateKernel(_id, name, &err);
        if (err != CL_SUCCESS)
            throw Exception("Could not create kernel. Wrong name?");
        
        Kernel* kernel = new Kernel (_controller, kernel_id);
        _kernels.push_back(kernel);
        return kernel;
    }
}
