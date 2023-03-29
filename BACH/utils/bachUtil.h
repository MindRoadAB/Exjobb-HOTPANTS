#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
};

struct Stamp {
  cl_double* stampData{};
  std::pair<cl_long, cl_long> stampCoords{};
  std::pair<cl_long, cl_long> stampSize{};
  std::vector<SubStamp> subStamps{};

  ~Stamp() { delete stampData; }
  Stamp(cl_double* stampData, std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize,
        const std::vector<SubStamp>& subStamps)
      : stampCoords{stampCoords}, stampSize{stampSize}, subStamps{subStamps} {
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = stampData[i];
    }
  }

  Stamp(const Stamp& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{other.subStamps} {
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
  }

  Stamp(Stamp&& other)
      : stampData{std::exchange(other.stampData, nullptr)},
        stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{std::move(other.subStamps)} {}

  Stamp& operator=(const Stamp& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = other.subStamps;
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = std::move(other.subStamps);
    this->stampData = std::exchange(other.stampData, nullptr);
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
