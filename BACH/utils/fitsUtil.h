#ifndef FITS_UTIL
#define FITS_UTIL

#include <CCfits/CCfits>
#include <CL/opencl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "argsUtil.h"

struct Image {
  std::string name;
  std::string path;
  std::pair<cl_long, cl_long> axis;
  std::vector<cl_double> data{};
  std::vector<bool> mask{};

  Image(const std::string n, std::pair<cl_long, cl_long> a = {0L, 0L},
        const std::string p = "res/")
      : name{n}, path{p}, axis{a}, data(this->size()), mask(this->size()) {}
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
        mask{std::move(other.maks)} {}

  Image& operator=(const Image& other) { return *this = Image{other}; }

  Image& operator=(Image&& other) {
    name = other.name;
    path = other.path;
    axis = other.axis;
    data = std::move(other.data);
    mask = std::move(other.mask);
    return *this;
  }

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

inline cl_int readImage(Image& input) {
  CCfits::FITS* pIn{};
  try {
    pIn = new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true);
  } catch(CCfits::FITS::CantOpen err) {
    std::cout << err.message() << std::endl;
    return -1;
  }
  CCfits::PHDU& img = pIn->pHDU();

  // Ifloat = -32, Idouble = -64
  cl_long type = img.bitpix();
  if(type != CCfits::Ifloat && type != CCfits::Idouble) {
    throw std::invalid_argument("fits image of type" + std::to_string(type) +
                                " is not supported.");
    return -1;
  }

  img.readAllKeys();

  input = Image{input.name, std::make_pair(img.axis(0), img.axis(1))};

  std::valarray<cl_double> temp(0.0, input.size());
  img.read(temp);
  input.data = std::vector<cl_double>{std::begin(temp), std::end(temp)};

  if(args.verbose) {
    std::cout << img << std::endl;
    std::cout << pIn->extension().size() << std::endl;
  }

  delete pIn;
  return 0;
}

inline cl_int writeImage(Image& img) {
  cl_long nAxis = 2;
  CCfits::FITS* pFits{};

  try {
    pFits = new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis,
                             img.axis_to_array());
  } catch(CCfits::FITS::CantCreate) {
    delete pFits;
    return -1;
  }

  cl_long fpixel(1);

  valarray<cl_double> data{&img.data[0], img.data.size()};

  pFits->pHDU().write(fpixel, data.size(), data);

  if(args.verbose) {
    std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl;
  }

  delete pFits;
  return 0;
}

#endif
