#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

#include "argsUtil.h"
#include "fitsUtil.h"

struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
  cl_double val;

  bool operator<(const SubStamp& other) { return val < other.val; }
  bool operator>(const SubStamp& other) { return val > other.val; }
};

struct Stamp {
  std::pair<cl_long, cl_long> stampCoords{};
  std::pair<cl_long, cl_long> stampSize{};
  std::vector<SubStamp> subStamps{};
  std::vector<cl_double> stampData{};

  Stamp(){};
  Stamp(std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize,
        const std::vector<SubStamp>& subStamps,
        const std::vector<cl_double>& stampData)
      : stampCoords{stampCoords},
        stampSize{stampSize},
        subStamps{subStamps},
        stampData{stampData} {}

  Stamp(const Stamp& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{other.subStamps},
        stampData{other.stampData} {}

  Stamp(Stamp&& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{std::move(other.subStamps)},
        stampData{std::move(other.stampData)} {}

  Stamp& operator=(const Stamp& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = other.subStamps;
    this->stampData = other.stampData;
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = std::move(other.subStamps);
    this->stampData = std::move(other.stampData);
    return *this;
  }
};

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

bool inImage(Image& image, int x, int y) {
  return !(x < 0 || x > image.axis.first || y < 0 || y > image.axis.second);
}

void checkSStamp(SubStamp& sstamp) {}

void findSStamps(Stamp& stamp, Image& image) {
  int foundSStamps = 0;
  cl_double floor, _tmp_;
  while(foundSStamps < args.maxSStamps) {
    int absx, absy, coords;
    _tmp_ = max(floor, sky + (args.threshHigh - sky) * ?);
    for(int x = 0; x < stamp.stampSize.first; x++) {
      absx = x + stamp.stampCoords.first;
      for(int y = 0; y < stamp.stampSize.second; y++) {
        absy = y + stamp.stampCoords.second;
        coords = x + (y * stamp.stampSize.second);
        if(image.masked(absx, absy) ||
           stamp.stampData[coords] > args.threshHigh ||
           stamp.stampData[coords] < args.threshLow) {  // TODO: add sky
          continue;
        }

        if(stamp.stampData[coords] > _tmp_) {  // good candidate found
          SubStamp s{std::make_pair(absx, absy), stamp.stampData[coords]};
          int kCoords;
          for(kx = absx - args.hSStampWidth; kx <= absx + args.hSStampWidth;
              kx++) {
            if(kx < 0 || kx > image.axis.first) continue;
            for(ky = absy - args.hSStampWidth; ky <= absy + args.hSStampWidth;
                ky++) {
              kCoords = kx + (ky * image.axis.first);
              if(ky < 0 || ky > image.axis.second || image.masked(kx, ky) ||
                 image[kCoords] < args.threshLow)  // TODO: add sky
                continue;

              if(image[kCoords] > args.threshHigh) {
                image.maskPix(kx, ky);
                continue;
              }

              if(image[kCoords] > s.val) {
                s.val = image[kCoords];
                s.subStampCoords = std::make_pair(kx, ky);
              }
            }
          }
          checkSStamp();
          stamp.subStamps.push_back(s);
          foundSStamps++;
          image.maskSStamp(s);
          if(args.verbose) std::cout << "Substamp added" << std::endl;
        }
        if(foundSStamps >= args.maxSStamps) break;
      }
      if(foundSStamps >= args.maxSStamps) break;
    }
    if() break;
  }

  if(foundSStamps == 0) {
    if(args.verbose)
      std::cout << "No suitable substamps found in stamp." << std::endl;
    return 1;
  }
  std::sort(stamp.subStamps.begin(), stamp.subStamps.end(),
            std::greater<SubStamp>());
}

bool notWithinThresh(SubStamp ss) {
  return (ss.val < args.threshLow || ss.val > args.threshHigh);
}

void identifySStamps(std::vector<Stamp>& stamps, Image& image) {
  for(auto s : stamps) {
    findSStamps(s, image);
    s.subStamps.erase(
        std::remove_if(s.subStamps.begin(), s.subStamps.end(), notWithinThresh),
        s.subStamps.end());
  }
}

#endif
