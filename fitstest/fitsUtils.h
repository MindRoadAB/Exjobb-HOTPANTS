#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <CCfits/CCfits>
#include <memory>
#include <string>

struct Image {
  std::string name;
  std::string path;
  std::string outName;
  std::valarray<float> data;
  /* std::map<string,CCfits::Keyword*>& keywords; */
  long axis[2];

  Image(const std::string inN) : name{inN}, path{"res/"}, outName{"out_" + inN}, data{}, axis{} {}
  Image(const std::string inN, const std::string inP, long* inA) : name{inN}, path{inP}, outName{"out_" + inN}, data{} {
    axis [0] = inA [0];
    axis [1] = inA [1];
  }

  std::string getFile (){
    return path+name;
  }

  std::string getOutFile (){
    return path+outName;
  }

};

int readImage(Image& input) {
  std::unique_ptr<CCfits::FITS> pIn(new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true));
  CCfits::PHDU& img = pIn->pHDU();

  img.readAllKeys();
  img.read(input.data);

  long ax1(img.axis(0));
  long ax2(img.axis(1));
  input.axis[0] = ax1;
  input.axis[1] = ax2;

  std::cout << img << std::endl;
  std::cout << pIn->extension().size() << std::endl;
  return 0;
}

int writeImage(Image& img) {
  long nAxis = 2;
  std::unique_ptr<CCfits::FITS> pFits{};

  try {
    pFits.reset(new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis, img.axis));
    std::cout << img.axis[0];
  } catch(CCfits::FITS::CantCreate) {
    return -1;
  }

  long nEl = std::accumulate(&img.axis[0], &img.axis[nAxis], 1, std::multiplies<long>());
  long fpixel(1);

  pFits->pHDU().write(fpixel, nEl, img.data);

  std::cout << pFits->pHDU() << std::endl;
  std::cout << pFits->extension().size() << std::endl;
  pFits.reset();

  return 0;
}
