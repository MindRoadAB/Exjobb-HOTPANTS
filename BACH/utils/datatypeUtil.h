#ifndef DATATYPE_UTIL
#define DATATYPE_UTIL

#include <CL/opencl.h>

#include <string>
#include <vector>

#include "argsUtil.h"


struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
  cl_double val;

  bool operator<(const SubStamp& other) const { return val < other.val; }
  bool operator>(const SubStamp& other) const { return val > other.val; }
};

struct StampStats {
  cl_double skyEst{};            // Mode of stamp
  cl_double fullWidthHalfMax{};  // Middle part value diff
};

struct Stamp {
  std::pair<cl_long, cl_long> stampCoords{};
  std::pair<cl_long, cl_long> stampSize{};
  std::vector<SubStamp> subStamps{};
  std::vector<cl_double> stampData{};
  StampStats stampStats{};

  Stamp(){};
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

  cl_double operator[](size_t index) { return stampData[index]; }
};

struct Image {
  std::string name;
  std::string path;
  std::pair<cl_long, cl_long> axis;
  std::vector<cl_double> data{};
  std::vector<bool> mask{};

  Image(const std::string n, std::pair<cl_long, cl_long> a = {0L, 0L},
        const std::string p = "res/")
      : name{n},
        path{p},
        axis{a},
        data(this->size()),
        mask(this->size(), false) {}

  Image(const std::string n, std::vector<cl_double> d,
        std::pair<cl_long, cl_long> a, std::vector<bool> m,
        const std::string p = "res/")
      : name{n}, path{p}, axis{a}, data{d}, mask{m} {}

  Image(const Image& other)
      : name{other.name},
        path{other.path},
        axis{other.axis},
        data{other.data},
        mask{other.mask} {}

  Image(Image&& other)
      : name{other.name},
        path{other.path},
        axis{other.axis},
        data{std::move(other.data)},
        mask{std::move(other.mask)} {}

  Image& operator=(const Image& other) { return *this = Image{other}; }

  Image& operator=(Image&& other) {
    name = other.name;
    path = other.path;
    axis = other.axis;
    data = std::move(other.data);
    mask = std::move(other.mask);
    return *this;
  }

  cl_double operator[](size_t index) { return data[index]; }

  std::string getFile() { return path + name; }

  std::string getFileName() {
    size_t lastI = name.find_last_of(".");
    return name.substr(0, lastI);
  }

  size_t size() { return (size_t)axis.first * axis.second; }

  std::string getOutFile() { return "!" + path + name; }

  long* axis_to_array() {
    static long tmpAx[2];
    tmpAx[0] = axis.first;
    tmpAx[1] = axis.second;
    long* ptr = tmpAx;
    return ptr;
  }

  bool masked(int x, int y) { return mask[x + (y * axis.first)]; }

  void maskPix(int x, int y) { mask[x + (y * axis.first)] = true; }

  void maskSStamp(SubStamp& sstamp) {
    for(int x = sstamp.subStampCoords.first - args.hSStampWidth;
        x <= sstamp.subStampCoords.first + args.hSStampWidth; x++) {
      if(x < 0 || x > axis.first) continue;
      for(int y = sstamp.subStampCoords.second - args.hSStampWidth;
          y <= sstamp.subStampCoords.second + args.hSStampWidth; y++) {
        if(y < 0 || y > axis.second) continue;
        this->maskPix(x, y);
      }
    }
  }
};

#endif
