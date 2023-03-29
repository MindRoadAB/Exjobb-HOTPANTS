#ifndef FITS_UTIL
#define FITS_UTIL

#include <CCfits/CCfits>
#include <memory>
#include <string>

#include "argsUtil.h"

struct Image {
  std::string name;
  std::string path;
  std::valarray<double> data;
  std::pair<long, long> axis;

  Image(const std::string n, const std::string p = "res/",
        std::valarray<double> v = {}, std::pair<long, long> a = {0L, 0L})
      : name{n}, path{p}, data{v}, axis{a} {}
  Image(const Image& other)
      : name{other.name},
        path{other.path},
        data{other.data},
        axis{other.axis} {}
  Image(Image&& other)
      : name{other.name},
        path{other.path},
        data{other.data},
        axis{other.axis} {}

  Image& operator=(const Image& other) { return *this = Image{other}; }

  Image& operator=(Image&& other) {
    name = other.name;
    path = other.path;
    data = other.data;
    axis = other.axis;
    return *this;
  }

  std::string getFile() { return path + name; }

  std::string getFileName() {
    size_t lastI = name.find_last_of(".");
    return name.substr(0, lastI);
  }

  std::string getOutFile() { return "!" + path + name; }

  long* axis_to_array() {
    static long tmpAx[2];
    tmpAx[0] = axis.first;
    tmpAx[1] = axis.second;
    long* ptr = tmpAx;
    return ptr;
  }
};

inline int readImage(Image& input) {
  CCfits::FITS* pIn{};
  try {
    pIn = new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true);
  } catch(CCfits::FITS::CantOpen err) {
    std::cout << err.message() << std::endl;
  }
  CCfits::PHDU& img = pIn->pHDU();

  // Ifloat = -32, Idouble = -64
  long type = img.bitpix();
  if(type != CCfits::Ifloat && type != CCfits::Idouble) {
    throw std::invalid_argument("fits image of type" + std::to_string(type) +
                                " is not supported.");
    return -1;
  }

  img.readAllKeys();
  img.read(input.data);

  long ax1(img.axis(0));
  long ax2(img.axis(1));
  input.axis.first = ax1;
  input.axis.second = ax2;

  if(args.verbose) {
    std::cout << img << std::endl;
    std::cout << pIn->extension().size() << std::endl;
  }

  delete pIn;
  return 0;
}

inline int writeImage(Image& img) {
  long nAxis = 2;
  CCfits::FITS* pFits{};

  try {
    pFits = new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis,
                             img.axis_to_array());
  } catch(CCfits::FITS::CantCreate) {
    delete pFits;
    return -1;
  }

  long fpixel(1);

  pFits->pHDU().write(fpixel, img.data.size(), img.data);

  if(args.verbose) {
    std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl;
  }

  delete pFits;
  return 0;
}

#endif
