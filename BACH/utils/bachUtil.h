#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

#include "fitsUtil.h"
#include "argsUtil.h"

struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
  cl_double val;
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

void findSStamps(Stamp& stamp) {
  for(int x = 0; x < stamp.stampSize.first; x++) {
    for(int y = 0; y < stamp.stampSize.second; y++) {

    }
  }
}

bool notWithinThresh(SubStamp ss) {
  return (ss.val < args.threshLow || ss.val > args.threshHigh); 
}

void identifySStamps(std::vector<Stamp>& stamps) {
  for(auto s : stamps) {
    findSStamps(s);
    std::remove_if(s.subStamps.begin(), s.subStamps.end(), notWithinThresh);
  }
}

#endif
