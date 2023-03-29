#ifndef BACH_UTIL
#define BACH_UTIL

#include <iostream>
#include <CL/opencl.hpp>

inline void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

#endif
