#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <CCfits/CCfits>
#include <memory>
#include <string>

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

  std::string getOutFile() { return "!" + path + name; }
};

int readImage(Image& input) {
  CCfits::FITS* pIn{};
  try {
    pIn = new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true);
  } catch(CCfits::FITS::CantOpen err) {
    std::cout << err.message() << std::endl;
  }
  CCfits::PHDU& img = pIn->pHDU();

  long type = img.bitpix();
  if(type != CCfits::Ifloat && type != CCfits::Idouble) {
    throw std::invalid_argument("fits image of type " + std::to_string(type) +
                                " is not supported.");
    return -1;
  }

  img.readAllKeys();
  img.read(input.data);

  long ax1(img.axis(0));
  long ax2(img.axis(1));
  input.axis.first = ax1;
  input.axis.second = ax2;

  /*   std::cout << img << std::endl;
    std::cout << pIn->extension().size() << std::endl; */

  delete pIn;
  return 0;
}

int writeImage(Image& img) {
  long nAxis = 2;
  CCfits::FITS* pFits{};

  try {
    long tmpAx[2];
    tmpAx[0] = img.axis.first;
    tmpAx[1] = img.axis.second;
    pFits = new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis, tmpAx);
  } catch(CCfits::FITS::CantCreate) {
    delete pFits;
    return -1;
  }

  long nEl = std::accumulate(&img.axis.first, &img.axis.second, 1,
                             std::multiplies<long>());
  long fpixel(1);

  pFits->pHDU().write(fpixel, nEl, img.data);

  /*  std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl; */

  delete pFits;
  return 0;
}
