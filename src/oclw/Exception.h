//
//  Exception.h
//  OCLW
//
//  Created by Srđan Rašić on 4/02/12.
//

#ifndef OCLW_Exception_h
#define OCLW_Exception_h

#include <exception>

namespace oclw {
    
    /*! Simple exception class.
     */
    class Exception : public std::exception {
    private:
        const char* _message;
        
    public:
        Exception (const char* message) : _message (message) {}
        
        virtual const char* what() const throw() {
            return _message;
        }
    };
}

#endif
