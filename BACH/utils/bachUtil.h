#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

struct Stamp {
  cl_double* stampData{};
  std::pair<cl_int, cl_int> stampCoords{};
  std::pair<cl_int, cl_int> stampSize{};
  std::pair<cl_int, cl_int> subStampCoords{};

  ~Stamp() { delete stampData; }
  Stamp(cl_double* stampData, std::pair<cl_int, cl_int> stampCoords,
        std::pair<cl_int, cl_int> stampSize,
        std::pair<cl_int, cl_int> subStampCoords)
      : stampCoords{stampCoords},
        stampSize{stampSize},
        subStampCoords{subStampCoords} {
    for(cl_int i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = stampData[i];
    }
  }

  Stamp(const Stamp& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStampCoords{other.subStampCoords} {
    for(cl_int i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
  }

  Stamp(Stamp&& other)
      : stampData{std::exchange(other.stampData, nullptr)},
        stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStampCoords{other.subStampCoords} {}

  Stamp& operator=(const Stamp& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStampCoords = other.subStampCoords;
    for(cl_int i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStampCoords = other.subStampCoords;
    stampData = std::exchange(other.stampData, nullptr);
    return *this;
  }
};

inline void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

#endif
