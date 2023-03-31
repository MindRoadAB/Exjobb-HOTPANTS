#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
};

struct Stamp {
  std::pair<cl_long, cl_long> stampCoords{};
  std::pair<cl_long, cl_long> stampSize{};
  std::vector<SubStamp> subStamps{};
  std::vector<cl_double> stampData{};

  Stamp(std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize,
        const std::vector<SubStamp>& subStamps,
        const std::vector<cl_double>& stampData)
      : stampCoords{stampCoords},
        stampSize{stampSize},
        subStamps{subStamps},
        stampData{stampData} {}

  Stamp(const Stamp& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{other.subStamps},
        stampData{other.stampData} {}

  Stamp(Stamp&& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{std::move(other.subStamps)},
        stampData{std::move(other.stampData)} {}

  Stamp& operator=(const Stamp& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = other.subStamps;
    this->stampData = other.stampData;
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = std::move(other.subStamps);
    this->stampData = std::move(other.stampData);
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
