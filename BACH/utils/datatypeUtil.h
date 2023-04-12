#ifndef DATATYPE_UTIL
#define DATATYPE_UTIL

#include <CL/opencl.h>

#include <iostream>
#include <string>
#include <vector>

#include "argsUtil.h"

struct Kernel {
  std::vector<cl_double> kernVec;

  inline void setKernVec() {
    int i = 0;
    for(int gauss = 0; gauss < args.dg.size(); gauss++) {
      for(int x = 0; x <= args.dg[gauss]; x++) {
        for(int y = 0; y <= args.dg[gauss] - x; y++) {
          kernVec[i] = 0.0;
          i++;
        }
      }
    }
  }
};

struct SubStamp {
  std::pair<cl_long, cl_long> imageCoords{};
  std::pair<cl_long, cl_long> stampCoords{};
  cl_double val;

  bool operator<(const SubStamp& other) const { return val < other.val; }
  bool operator>(const SubStamp& other) const { return val > other.val; }
};

struct StampStats {
  cl_double skyEst{};  // Mode of stamp
  cl_double fwhm{};    // Middle part value diff (full width half max)
};

struct Stamp {
  std::pair<cl_long, cl_long> coords{};
  std::pair<cl_long, cl_long> size{};
  std::vector<SubStamp> subStamps{};
  std::vector<cl_double> data{};
  StampStats stats{};

  Stamp(){};
  Stamp(std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize,
        const std::vector<SubStamp>& subStamps,
        const std::vector<cl_double>& stampData)
      : coords{stampCoords},
        size{stampSize},
        subStamps{subStamps},
        data{stampData} {}

  Stamp(const Stamp& other)
      : coords{other.coords},
        size{other.size},
        subStamps{other.subStamps},
        data{other.data} {}

  Stamp(Stamp&& other)
      : coords{other.coords},
        size{other.size},
        subStamps{std::move(other.subStamps)},
        data{std::move(other.data)} {}

  Stamp& operator=(const Stamp& other) {
    this->coords = other.coords;
    this->size = other.size;
    this->subStamps = other.subStamps;
    this->data = other.data;
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->coords = other.coords;
    this->size = other.size;
    this->subStamps = std::move(other.subStamps);
    this->data = std::move(other.data);
    return *this;
  }

  cl_double operator[](size_t index) { return data[index]; }
};

struct Image {
  std::string name;
  std::string path;
  std::pair<cl_long, cl_long> axis;

  enum masks { nan, badInput, badPixel, psf };

 private:
  std::vector<cl_double> data{};
  std::vector<bool> nanMask{};
  std::vector<bool> badInputMask{};
  std::vector<bool> badPixelMask{};
  std::vector<bool> psfMask{};

 public:
  Image(const std::string n, std::pair<cl_long, cl_long> a = {0L, 0L},
        const std::string p = "res/")
      : name{n},
        path{p},
        axis{a},
        data(this->size()),
        nanMask(this->size(), false),
        badInputMask(this->size(), false),
        badPixelMask(this->size(), false),
        psfMask(this->size(), false) {}

  Image(const std::string n, std::vector<cl_double> d,
        std::pair<cl_long, cl_long> a, const std::string p = "res/")
      : name{n},
        path{p},
        axis{a},
        data{d},
        nanMask(this->size(), false),
        badInputMask(this->size(), false),
        badPixelMask(this->size(), false),
        psfMask(this->size(), false) {}

  Image(const Image& other)
      : name{other.name},
        path{other.path},
        axis{other.axis},
        data{other.data},
        nanMask(other.nanMask),
        badInputMask(other.badInputMask),
        badPixelMask(other.badPixelMask),
        psfMask(other.psfMask) {}

  Image(Image&& other)
      : name{other.name},
        path{other.path},
        axis{other.axis},
        data{std::move(other.data)},
        nanMask(std::move(other.nanMask)),
        badInputMask(std::move(other.badInputMask)),
        badPixelMask(std::move(other.badPixelMask)),
        psfMask(std::move(other.psfMask)) {}

  Image& operator=(const Image& other) {
    name = other.name;
    path = other.path;
    axis = other.axis;
    data = other.data;
    nanMask = other.nanMask;
    badInputMask = other.badInputMask;
    badPixelMask = other.badPixelMask;
    psfMask = other.psfMask;
    return *this;
  }

  Image& operator=(Image&& other) {
    name = other.name;
    path = other.path;
    axis = other.axis;
    data = std::move(other.data);
    nanMask = std::move(other.nanMask);
    badInputMask = std::move(other.badInputMask);
    badPixelMask = std::move(other.badPixelMask);
    psfMask = std::move(other.psfMask);
    return *this;
  }

  cl_double* operator&() { return &data[0]; }

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

  bool masked(int x, int y, masks m) {
    switch(m) {
      case nan:
        return nanMask[x + (y * axis.first)];
        break;
      case badInput:
        return badInputMask[x + (y * axis.first)];
        break;
      case badPixel:
        return badPixelMask[x + (y * axis.first)];
        break;
      case psf:
        return psfMask[x + (y * axis.first)];
        break;
      default:
        std::cout << "Error: Not caught by the switch case" << std::endl;
        exit(1);
    }
  }

  void maskPix(int x, int y, masks m) {
    switch(m) {
      case nan:
        nanMask[x + (y * axis.first)] = true;
        return;
      case badInput:
        badInputMask[x + (y * axis.first)] = true;
        return;
      case badPixel:
        badPixelMask[x + (y * axis.first)] = true;
        return;
      case psf:
        psfMask[x + (y * axis.first)] = true;
        return;
      default:
        std::cout << "Error: Not caught by the switch case" << std::endl;
        exit(1);
    }
  }

  void maskSStamp(SubStamp& sstamp, masks m) {
    for(int x = sstamp.imageCoords.first - args.hSStampWidth;
        x < sstamp.imageCoords.first + args.hSStampWidth; x++) {
      if(x < 0 || x >= axis.first) continue;
      for(int y = sstamp.imageCoords.second - args.hSStampWidth;
          y < sstamp.imageCoords.second + args.hSStampWidth; y++) {
        if(y < 0 || y >= axis.second) continue;
        this->maskPix(x, y, m);
      }
    }
  }

  void maskAroundPix(int inX, int inY, masks m) {
    for(int x = inX - args.hSStampWidth; x < inX + args.hSStampWidth; x++) {
      if(x < 0 || x >= axis.first) continue;
      for(int y = inY - args.hSStampWidth; y < inY + args.hSStampWidth; y++) {
        if(y < 0 || y >= axis.second) continue;
        this->maskPix(x, y, m);
      }
    }
  }
};

#endif
