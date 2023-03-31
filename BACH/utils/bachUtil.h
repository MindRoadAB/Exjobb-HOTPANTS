#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <iostream>
#include <utility>

#include "fitsUtil.h"

struct SubStamp {
  std::pair<cl_long, cl_long> subStampCoords{};
};

struct Stamp {
  cl_double* stampData{};
  std::pair<cl_long, cl_long> stampCoords{};
  std::pair<cl_long, cl_long> stampSize{};
  std::vector<SubStamp> subStamps{};

  ~Stamp() { delete stampData; }
  Stamp() {};
  Stamp(cl_double* stampData, std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize,
        const std::vector<SubStamp>& subStamps)
      : stampCoords{stampCoords}, stampSize{stampSize}, subStamps{subStamps} {
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = stampData[i];
    }
  }

  Stamp(const Stamp& other)
      : stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{other.subStamps} {
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
  }

  Stamp(Stamp&& other)
      : stampData{std::exchange(other.stampData, nullptr)},
        stampCoords{other.stampCoords},
        stampSize{other.stampSize},
        subStamps{std::move(other.subStamps)} {}

  Stamp& operator=(const Stamp& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = other.subStamps;
    for(cl_long i = 0; i < stampSize.first * stampSize.second; i++) {
      this->stampData[i] = other.stampData[i];
    }
    return *this;
  }

  Stamp& operator=(Stamp&& other) {
    this->stampCoords = other.stampCoords;
    this->stampSize = other.stampSize;
    this->subStamps = std::move(other.subStamps);
    this->stampData = std::exchange(other.stampData, nullptr);
    return *this;
  }
};

inline void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

std::vector<Stamp>& createStamps(Image& img, int w, int h) {
  std::vector<Stamp> stamps(args.stampsx * args.stampsy, Stamp{});
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

      cl_double* tmp = new cl_double[stampw * stamph];
      for(int x = 0; x < stampw; x++) {
        for(int y = 0; y < stamph; y++) {
          tmp[x + (y * stampw)] = img.data[(startx + x) + ((starty + y) * w)];
        }
      }

      stamps[i + j * args.stampsx].stampData = tmp;
      stamps[i + j * args.stampsx].stampCoords = {startx, starty};
      stamps[i + j * args.stampsx].stampSize = {stampw, stamph};
    }
  } 
  return stamps;
}

#endif
