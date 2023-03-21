#include <iostream>

#include "argsUtil.h"
#include "fitsUtil.h"

int main(int argc, char* argv[]) {
  CCfits::FITS::setVerboseMode(true);

  try {
    getArguments(argc, argv);
  } catch(const std::invalid_argument& err) {
    std::cout << err.what() << '\n';
    return 1;
  }

  Image templateImg{args.templateName};
  Image scienceImg{args.scienceName};

  readImage(templateImg);

  Image outImg{args.outName, args.outPath, templateImg.data, templateImg.axis};
  writeImage(outImg);

  return 0;
}