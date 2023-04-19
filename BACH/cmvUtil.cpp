#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "utils/bachUtil.h"

void createB(Stamp& s, Image& img) {
  /* Does Equation 2.13 which create the right side of the Equation Ma=B */
  if(args.verbose) std::cout << "Creating B..." << std::endl;

  s.B.emplace_back();
  int ssx = s.subStamps[0].imageCoords.first;
  int ssy = s.subStamps[0].imageCoords.second;

  for(int i = 0; i < args.nPSF; i++) {
    cl_double p0 = 0.0;
    for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
      for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
        int k = x + args.hSStampWidth +
                (args.hSStampWidth * 2) * (y + args.hSStampWidth);
        int imgIndex = x + ssx + (y + ssy) * img.axis.first;
        p0 += s.W[i][k] * img[imgIndex];
      }
    }
    s.B.push_back(p0);
  }

  cl_double q = 0.0;
  for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
    for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
      int k = x + args.hSStampWidth +
              (args.hSStampWidth * 2) * (y + args.hSStampWidth);
      int imgIndex = x + ssx + (y + ssy) * img.axis.first;
      q += s.W[args.nPSF][k] * img[imgIndex];
    }
  }
  s.B.push_back(q);
}

void convStamp(Stamp& s, Image& img, Kernel& k, int n, int odd) {
  /*
   * Fills a Stamp with a convolved version (using only gaussian basis functions
   * without amlitude) of the area around its selected substamp.
   *
   * This can result in nan values but which should be handeld later.
   */

  if(args.verbose) std::cout << "Convolving stamp..." << std::endl;

  s.W.emplace_back();
  cl_long ssx = s.subStamps[0].imageCoords.first;
  cl_long ssy = s.subStamps[0].imageCoords.second;

  std::vector<cl_double> tmp{};

  // Convolve Image with filterY taking pixels in a (args.hSStampWidth +
  // args.hKernelWidth) area around a substamp.
  for(int i = ssx - args.hSStampWidth - args.hKernelWidth;
      i <= ssx + args.hSStampWidth + args.hKernelWidth; i++) {
    for(int j = ssy - args.hSStampWidth; j <= ssy + args.hSStampWidth; j++) {
      tmp.push_back(0.0);

      for(int y = args.hKernelWidth; y <= args.hKernelWidth; y++) {
        int imgIndex = i + (j + y) * img.axis.first;
        tmp.back() += img[imgIndex] * k.filterY[n][args.hKernelWidth - y];
      }
    }
  }

  // Convolve Image with filterX, image data already there.
  for(int j = -args.hSStampWidth; j < args.hSStampWidth; j++) {
    for(int i = -args.hSStampWidth; i < args.hSStampWidth; i++) {
      int index =
          i + args.hSStampWidth + (j + args.hSStampWidth) * args.fSStampWidth;
      s.W[n].push_back(0.0);
      for(int x = args.hKernelWidth; x <= args.hKernelWidth; x++) {
        s.W.back().back() += tmp[index] * k.filterX[n][args.hKernelWidth - x];
      }
    }
  }

  // Removes n = 0 vector from all odd vectors in s.W
  // TODO: Find out why this is done.....
  if(odd) {
    for(int i = 0; i < args.fSStampWidth * args.fSStampWidth; i++)
      s.W[n][i] -= s.W[0][i];
  }
}

void cutSStamp(SubStamp& ss, Image& img) {
  /* Store the original image data around the substamp in said substamp */
  if(args.verbose) std::cout << "Cutting substamp..." << std::endl;

  for(int y = 0; y < args.fSStampWidth; y++) {
    int imgY = ss.imageCoords.second + y - args.hSStampWidth;

    for(int x = 0; x < args.fSStampWidth; x++) {
      int imgX = ss.imageCoords.first + x - args.hSStampWidth;

      ss.data.push_back(img[imgX + imgY * img.axis.first]);
      ss.sum += img.masked(imgX, imgY, Image::badInput, Image::nan)
                    ? 0.0
                    : abs(img[imgX + imgY * img.axis.first]);
    }
  }
}

void fillStamp(Stamp& s, Image& tImg, Image& sImg, Kernel& k) {
  /* Fills Substamp with gaussian basis convolved images around said substamp
   * and claculates CMV.
   */
  if(args.verbose) std::cout << "Filling stamp..." << std::endl;
  if(s.subStamps.empty()) {
    if(args.verbose)
      std::cout << "No eligable substamps, stamp rejected" << std::endl;
    return;
  }

  int nvec = 0;
  for(int g = 0; g < cl_int(args.dg.size()); g++) {
    for(int x = 0; x <= args.dg[g]; x++) {
      for(int y = 0; y <= args.dg[g] - x; y++) {
        int odd = 0;

        cl_double dx = (x / 2.0) * 2 - x;
        cl_double dy = (y / 2.0) * 2 - y;
        if(dx == 0 && dy == 0 && nvec > 0) odd = 1;

        convStamp(s, tImg, k, nvec, odd);
        if(args.verbose) std::cout << "Stamp convolved" << std::endl;
        nvec++;
      }
    }
  }

  cutSStamp(s.subStamps[0], sImg);

  cl_long ssx = s.subStamps[0].imageCoords.first;
  cl_long ssy = s.subStamps[0].imageCoords.second;

  for(int j = 0; j <= args.backgroundOrder; j++) {
    for(int k = 0; k <= args.backgroundOrder - j; k++) {
      s.W.emplace_back();
    }
  }

  for(int x = ssx - args.hSStampWidth; x < ssx + args.hSStampWidth; x++) {
    for(int y = ssy - args.hSStampWidth; y < ssy + args.hSStampWidth; y++) {
      cl_double ax = 1.0;
      cl_int nBGVec = 0;
      for(int j = 0; j <= args.backgroundOrder; j++) {
        cl_double ay = 1.0;
        for(int k = 0; k <= args.backgroundOrder - j; k++) {
          s.W[args.nPSF + nBGVec++].push_back(ax * ay);
          ay *= (y - tImg.axis.second * 0.5) / tImg.axis.second * 0.5;
        }
        ax *= (x - tImg.axis.first * 0.5) / tImg.axis.first * 0.5;
      }
    }
  }

  s.createQ();  // TODO: is name accurate?
  createB(s, sImg);
}
