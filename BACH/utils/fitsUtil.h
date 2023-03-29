#ifndef FITS_UTIL
#define FITS_UTIL

#include <CCfits/CCfits>
#include <CL/opencl.hpp>
#include <memory>
#include <string>

#include "argsUtil.h"

struct Image {
  std::string name;
  std::string path;
  std::pair<cl_long, cl_long> axis;
  cl_double* data;

  ~Image() { delete data; }
  Image(const std::string n, std::pair<cl_long, cl_long> a = {0L, 0L},
        const std::string p = "res/")
      : name{n}, path{p}, axis{a} {
    data = new cl_double[this->size()];
  }
  Image(const std::string n, cl_double* d, std::pair<cl_long, cl_long> a,
        const std::string p = "res/")
      : name{n}, path{p}, axis{a}, data{d} {}

  Image(const Image& other)
      : name{other.name}, path{other.path}, axis{other.axis} {
    delete data;
    data = new cl_double[this->size()];
    for(size_t i = 0; i < this->size(); i++) {
      data[i] = other.data[i];
    }
  }
  Image(Image&& other)
      : name{other.name},
        path{other.path},
        axis{other.axis},
        data{std::exchange(other.data, nullptr)} {}

  Image& operator=(const Image& other) { return *this = Image{other}; }

  Image& operator=(Image&& other) {
    name = other.name;
    path = other.path;
    axis = other.axis;
    data = std::exchange(other.data, nullptr);
    return *this;
  }

  void updateImage(std::pair<cl_long, cl_long> a, cl_double* d = nullptr) {
    axis = a;
    delete data;
    if(d != nullptr) {
      data = d;
    } else {
      data = new cl_double[this->size()];
    }
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

  input.updateImage(std::make_pair(img.axis(0), img.axis(1)));

  std::valarray<cl_double> temp(0.0, input.size());
  img.read(temp);
  for(size_t i = 0; i < input.size(); i++) {
    input.data[i] = temp[i];
  }

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

  valarray<cl_double> data{img.data, img.size()};

  pFits->pHDU().write(fpixel, data.size(), data);

  if(args.verbose) {
    std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl;
  }

  delete pFits;
  return 0;
}

#endif
