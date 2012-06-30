//
//  OpenCL.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

/** Ensures correct header inclusion on
    different platforms.
 */

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif