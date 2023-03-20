#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <CCfits/CCfits>
#include <memory>
#include <string>

int readImage(const std::string name, std::valarray<unsigned long>& contents) {
  std::auto_ptr<CCfits::FITS> pIn(new CCfits::FITS(name, CCfits::RWmode::Read, true));
  CCfits::PHDU& img = pIn->pHDU();

  img.readAllKeys();
  img.read(contents);

  long ax1(img.axis(0));
  long ax2(img.axis(1));

  // print image
  for(long i = 0; i < ax2; i += 10) {
    std::ostream_iterator<short> c(std::cout, "\t");
    std::copy(&contents[i*ax1], &contents[(i+1)*ax1-1],c);
    std::cout << '\n';
  }

  return 0;
}

int writeImage(long *nAxes, const std::string name, std::valarray<unsigned long >& contents) {
  long nAxis = 2;
  std::auto_ptr<CCfits::FITS> pFits(0);

  try {
    pFits.reset(new CCfits::FITS(name, USHORT_IMG, nAxis, nAxes));
  } catch(CCfits::FITS::CantCreate) {
    return -1;
  }

  long nEl = std::accumulate(&nAxes[0], &nAxes[nAxis], 1, std::multiplies<long>());

  std::vector<long> extAx(nAxis, nAxes[0]); // might need to change
  string newName("NEW-EXTENSION");
  CCfits::ExtHDU* imgExt = pFits->addImage(newName, FLOAT_IMG, extAx);

  long nExtEl = std::accumulate(extAx.begin(), extAx.end(), 1, std::multiplies<long>());

  long fpixel(0);
  imgExt->write(fpixel, nExtEl, contents);
  return 0;
}

int main() {
  std::valarray<unsigned long> templImg;
  readImage("template2.fits", templImg);

/*   CCfits::FITS::setVerboseMode(true); */
  return 0;
}
