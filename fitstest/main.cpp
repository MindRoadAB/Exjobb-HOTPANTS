#include "fitsUtils.h"
#include <iostream>

int main() {
  Image templImg{"template2.fits"};
  CCfits::FITS::setVerboseMode(true);\
  try {
    readImage(templImg);
  } catch (const std::invalid_argument& e){
    std::cout << e.what() << '\n';
    return -1;
  }
  if(writeImage(templImg) != 0)
    return -1;
 
  Image written{"out_template2.fits"};
  readImage(written);

 for(size_t i = 0; i < templImg.data.size(); i++){
    if(templImg.data[i] != written.data[i])
      std::cout << "not equal" << std::endl;
  }

  return 0;
}
