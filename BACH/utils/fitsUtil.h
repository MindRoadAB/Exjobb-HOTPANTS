#ifndef FITS_UTIL
#define FITS_UTIL

#include <CCfits/CCfits>
#include <CL/opencl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "argsUtil.h"
#include "datatypeUtil.h"

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
