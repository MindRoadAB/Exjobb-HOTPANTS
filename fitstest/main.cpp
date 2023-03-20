#include "fitsUtils.h"

int main() {
  Image templImg{"template2.fits"};
  CCfits::FITS::setVerboseMode(true);
  readImage(templImg);
  writeImage(templImg);
 
  Image written{"out_template2.fits"};
  readImage(written);

 for(size_t i = 0; i < templImg.data.size(); i++){
    if(templImg.data[i] != written.data[i])
      std::cout << "not equal" << std::endl;
  }

  return 0;
}
