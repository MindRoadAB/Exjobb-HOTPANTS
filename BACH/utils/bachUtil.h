#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "argsUtil.h"
#include "datatypeUtil.h"

inline void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

inline void createStamps(Image& img, std::vector<Stamp>& stamps, int w, int h) {
  for(int i = 0; i < args.stampsx; i++) {
    for(int j = 0; j < args.stampsy; j++) {
      int stampw = w / args.stampsx;
      int stamph = h / args.stampsy;
      int startx = i * stampw;
      int starty = j * stamph;
      int stopx = startx + stampw;
      int stopy = starty + stamph;
      if(i == args.stampsx - 1) {
        stopx = w;
        stampw = stopx - startx;
      }
      if(j == args.stampsy - 1) {
        stopy = h;
        stamph = stopy - starty;
      }

      for(int x = 0; x < stampw; x++) {
        for(int y = 0; y < stamph; y++) {
          cl_double tmp = img.data[(startx + x) + ((starty + y) * w)];
          stamps[i + j * args.stampsx].stampData.push_back(tmp);
        }
      }

      stamps[i + j * args.stampsx].stampCoords = {startx, starty};
      stamps[i + j * args.stampsx].stampSize = {stampw, stamph};
    }
  }
}

inline bool inImage(Image& image, int x, int y) {
  return !(x < 0 || x > image.axis.first || y < 0 || y > image.axis.second);
}

inline void checkSStamp(SubStamp& sstamp) {
  std::cout << "checking sub-stamp..." << std::endl;
}

inline cl_int findSStamps(Stamp& stamp, Image& image) {
  std::cout << "Looking for sub-stamps in stamp..." << std::endl;
  int foundSStamps = 0;
  cl_double floor = stamp.stampStats.skyEst +
                    args.threshKernFit * stamp.stampStats.fullWidthHalfMax;
  cl_double _tmp_;
  cl_double dfrac = 0.9;
  while(foundSStamps < args.maxSStamps) {
    int absx, absy, coords;
    _tmp_ = std::max(floor, stamp.stampStats.skyEst +
                                (args.threshHigh - stamp.stampStats.skyEst) *
                                    dfrac);  // TODO: why is dfrac == 0.9?
    for(int x = 0; x < stamp.stampSize.first; x++) {
      absx = x + stamp.stampCoords.first;
      for(int y = 0; y < stamp.stampSize.second; y++) {
        absy = y + stamp.stampCoords.second;
        coords = x + (y * stamp.stampSize.second);

        if(image.masked(absx, absy) ||
           stamp.stampData[coords] > args.threshHigh ||
           (stamp.stampData[coords] - stamp.stampStats.skyEst) *
                   (1.0 / stamp.stampStats.fullWidthHalfMax) <
               args.threshKernFit) {
          continue;
        }

        if(stamp.stampData[coords] > _tmp_) {  // good candidate found
          SubStamp s{std::make_pair(absx, absy), stamp.stampData[coords]};
          int kCoords;
          for(int kx = absx - args.hSStampWidth; kx < absx + args.hSStampWidth;
              kx++) {
            if(kx < 0 || kx >= image.axis.first) {
              /* std::cout << "continue 1..." << std::endl; */
              continue;
            }
            for(int ky = absy - args.hSStampWidth;
                ky < absy + args.hSStampWidth; ky++) {
              kCoords = kx + (ky * image.axis.first);
              bool inKernFit =
                  (stamp.stampData[coords] - stamp.stampStats.skyEst) *
                      (1.0 / stamp.stampStats.fullWidthHalfMax) <
                  args.threshKernFit;
              if(ky < 0 || ky >= image.axis.second || image.masked(kx, ky) ||
                 inKernFit) {
                /* std::cout << "continue 2..." << std::endl; */
                continue;
              }

              if(image.data[kCoords] > args.threshHigh) {
                image.maskPix(kx, ky);
                /* std::cout << "continue 3..." << std::endl; */
                continue;
              }

              if(image[kCoords] > s.val) {
                s.val = image[kCoords];
                s.subStampCoords = std::make_pair(kx, ky);
              }
            }
          }

          checkSStamp(s);
          stamp.subStamps.push_back(s);
          foundSStamps++;
          image.maskSStamp(s);
          if(args.verbose) std::cout << "Substamp added" << std::endl;
        }
        if(foundSStamps >= args.maxSStamps) break;
      }
      if(foundSStamps >= args.maxSStamps) break;
    }
    // Compare _tmp_ against a floor value;
    if(_tmp_ == floor) break;
    dfrac -= 0.2;
  }

  if(foundSStamps == 0) {
    if(args.verbose)
      std::cout << "No suitable substamps found in stamp." << std::endl;
    return 1;
  }
  std::sort(stamp.subStamps.begin(), stamp.subStamps.end(),
            std::greater<SubStamp>());
  return 0;
}

inline bool notWithinThresh(SubStamp ss) {
  return (ss.val < args.threshLow || ss.val > args.threshHigh);
}

inline void sigmaClip(std::vector<cl_double>& data, cl_double& mean,
                      cl_double& stdDev) {
  /* Does sigma clipping on data to provide the mean and stdDev of said
   * data
   */
  if(data.empty()) {
    std::cout << "Cannot send in empty vector to Sigma Clip" << std::endl;
    exit(1);
  }

  size_t currNPoints = 0;
  size_t prevNPoints = data.size();
  std::vector<bool> intMask(data.size(), false);

  // Do three times or a stable solution has been found.
  for(int i = 0; (i < 3) && (currNPoints != prevNPoints); i++) {
    currNPoints = prevNPoints;
    mean = 0;
    stdDev = 0;

    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        mean += data[i];
        stdDev += data[i] * data[i];
      }
    }

    if(prevNPoints > 1) {
      mean = mean / prevNPoints;
      stdDev = stdDev - prevNPoints * mean * mean;
      stdDev = std::sqrt(stdDev / double(prevNPoints - 1));
    } else {
      std::cout << "prevNPoints is: " << prevNPoints
                << "Needs to be greater than 1" << std::endl;
      exit(1);
    }

    prevNPoints = 0;
    double invStdDev = 1.0 / stdDev;
    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        // Doing the sigmaClip
        if(abs(data[i] - mean) * invStdDev > args.sigClipAlpha) {
          intMask[i] = true;
        } else {
          prevNPoints++;
        }
      }
    }
  }
}

inline void calcStats(Stamp& stamp, Image& image) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   * TODO: Fix Masking, very not correct. Also mask bad pixels found in here.
   * TODO: Compare results on same stamp on this and old version.
   */

  cl_double median, lfwhm, sum;  // temp for now

  std::vector<cl_double> values{};
  std::vector<cl_int> bins(256, 0);

  cl_int nValues = 100;
  cl_double upProc = 0.9;
  cl_double midProc = 0.5;
  cl_int numPix = stamp.stampSize.first * stamp.stampSize.second;

  if(numPix < nValues) {
    std::cout << "Not enough pixels in a stamp" << std::endl;
    exit(1);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<cl_int> randGenX(0, stamp.stampSize.first - 1);
  std::uniform_int_distribution<cl_int> randGenY(0, stamp.stampSize.second - 1);

  // Stop after randomly having selected a pixel numPix times.
  for(int i = 0; (i < numPix) && (values.size() < nValues); i++) {
    int randX = randGenX(gen);
    int randY = randGenY(gen);

    // Random pixel in stamp in stamp coords.
    cl_int indexS = randX + randY * stamp.stampSize.first;

    // Random pixel in stamp in Image coords.
    cl_int indexI = (randX + stamp.stampCoords.first) +
                    (randY + stamp.stampCoords.second) * stamp.stampSize.first;

    if(image.mask[indexI] || stamp[indexS] < 0) {
      continue;
    }

    values.push_back(stamp[indexS]);
  }

  std::sort(begin(values), end(values));

  // Width of a histogram bin.
  cl_double binSize = (values[(int)(upProc * values.size())] -
                       values[(int)(midProc * values.size())]) /
                      (cl_double)nValues;
  std::cout << "binSize: " << binSize << std::endl;

  // Value of lowest bin.
  cl_double lowerBinVal =
      values[(int)(midProc * values.size())] - (128.0 * binSize);

  // Contains all good Pixels in the stamp, aka not masked.
  std::vector<cl_double> maskedStamp{};
  for(int x = 0; x < stamp.stampSize.first; x++) {
    for(int y = 0; y < stamp.stampSize.second; y++) {
      // Pixel in stamp in stamp coords.
      cl_int indexS = x + y * stamp.stampSize.first;

      // Pixel in stamp in Image coords.
      cl_int indexI = (x + stamp.stampCoords.first) +
                      (y + stamp.stampCoords.second) * stamp.stampSize.first;

      if(!image.mask[indexI] && stamp[indexS] >= 0) {
        maskedStamp.push_back(stamp[indexS]);
      }
    }
  }

  // sigma clip of maskedStamp to get mean and sd.
  cl_double mean, stdDev, invStdDev;
  sigmaClip(maskedStamp, mean, stdDev);
  invStdDev = 1.0 / stdDev;

  if(args.verbose) {
    std::cout << "Mean: " << mean << " stdDev: " << stdDev << std::endl;
  }

  int attempts = 0;
  cl_int okCount = 0;
  cl_double sumBins = 0.0;
  cl_double sumExpect = 0.0;
  cl_double lower, upper;
  while(true) {
    if(attempts >= 5) {
      std::cout << "Creation of histogram unsuccessful after 5 attempts.";
      exit(1);
    }

    std::fill(bins.begin(), bins.end(), 0);
    okCount = 0;
    sum = 0.0;
    sumBins = 0.0;
    sumExpect = 0.0;
    for(int x = 0; x < stamp.stampSize.first; x++) {
      for(int y = 0; y < stamp.stampSize.second; y++) {
        // Pixel in stamp in stamp coords.
        cl_int indexS = x + y * stamp.stampSize.first;

        // Pixel in stamp in Image coords.
        cl_int indexI = (x + stamp.stampCoords.first) +
                        (y + stamp.stampCoords.second) * stamp.stampSize.first;

        if(image.mask[indexI] || stamp[indexS] < 0) {
          continue;
        }

        if((abs(stamp[indexS] - mean) * invStdDev) > args.sigClipAlpha) {
          continue;
        }

        int index = std::clamp(
            (int)std::floor((stamp[indexS] - lowerBinVal) / binSize), 0, 255);

        bins[index]++;
        sum += abs(stamp[indexS]);
        okCount++;
      }
    }

    if(okCount == 0 || binSize == 0.0) {
      std::cout << "No good pixels or variation in pixels." << std::endl;
      exit(1);
    }

    cl_double maxDens = 0.0;
    int lowerIndex, upperIndex, maxIndex;
    for(lowerIndex = upperIndex = 1; upperIndex < 255;
        sumBins -= bins[lowerIndex++]) {
      while(sumBins < okCount / 10.0 && upperIndex < 255) {
        sumBins += bins[upperIndex++];
      }
      if(sumBins / (upperIndex - lowerIndex) > maxDens) {
        maxDens = sumBins / (upperIndex - lowerIndex);
        maxIndex = lowerIndex;
      }
    }
    if(maxIndex < 0 || maxIndex > 255) maxIndex = 0;

    sumBins = 0.0;
    for(int i = maxIndex; sumBins < okCount / 10.0 && i < 255; i++) {
      sumBins += bins[i];
      sumExpect += i * bins[i];
    }
    cl_double modeBin = sumExpect / sumBins + 0.5;
    stamp.stampStats.skyEst = lowerBinVal + binSize * (modeBin - 1.0);
    if(args.verbose)
      std::cout << "Mode: " << stamp.stampStats.skyEst << std::endl;

    lower = okCount * 0.25;
    upper = okCount * 0.75;
    sumBins = 0.0;
    int i = 0;
    for(; sumBins < lower; sumBins += bins[i++])
      ;
    lower = i - (sumBins - lower) / bins[i - 1];
    for(; sumBins < upper; sumBins += bins[i++])
      ;
    upper = i - (sumBins - upper) / bins[i - 1];

    if(lower < 1.0 || upper > 255.0) {
      if(args.verbose) {
        std::cout << "lower is: " << lower << ", upper is: " << upper
                  << std::endl;
        std::cout << "Expanding bin size..." << std::endl;
      }
      lowerBinVal -= 128.0 * binSize;
      binSize *= 2;
      attempts++;
    } else if(upper - lower < 40.0) {
      if(args.verbose) {
        std::cout << "lower is: " << lower << ", upper is: " << upper
                  << std::endl;
        std::cout << "Shrinking bin size..." << std::endl;
      }
      binSize /= 3.0;
      lowerBinVal = stamp.stampStats.skyEst - 128.0 * binSize;
      attempts++;
    } else
      break;
  }
  stamp.stampStats.fullWidthHalfMax = binSize * (upper - lower) / args.iqRange;
  if(args.verbose)
    std::cout << "fwhm: " << stamp.stampStats.fullWidthHalfMax << std::endl;
  int i = 0;
  for(i = 0, sumBins = 0; sumBins < okCount / 2.0; sumBins += bins[i++])
    ;
  median = i - (sumBins - okCount / 2.0) / bins[i - 1];
  lfwhm = binSize * (median - lower) * 2.0 / args.iqRange;
  median = lowerBinVal + binSize * (median - 1.0);
}

inline void identifySStamps(std::vector<Stamp>& stamps, Image& image) {
  std::cout << "Identifying for sub-stamps..." << std::endl;
  for(auto& s : stamps) {
    calcStats(s, image);
    findSStamps(s, image);
    s.subStamps.erase(
        std::remove_if(s.subStamps.begin(), s.subStamps.end(), notWithinThresh),
        s.subStamps.end());
  }
}

#endif
