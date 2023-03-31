#ifndef ARG_UTIL
#define ARG_UTIL

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

struct Arguments {
  std::string templateName;
  std::string scienceName;
  std::string outName = "diff.fits";

  std::string inputPath = "res/";
  std::string outPath = "out/";

  int stampsx = 10;
  int stampsy = 10;

  bool verbose = false;
};

inline char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

inline bool cmdOptionExists(char** begin, char** end,
                            const std::string& option) {
  return std::find(begin, end, option) != end;
}

inline Arguments args{};

inline void getArguments(int argc, char* argv[]) {
  if(cmdOptionExists(argv, argv + argc, "-o")) {
    args.outName = getCmdOption(argv, argv + argc, "-o");
  }

  if(cmdOptionExists(argv, argv + argc, "-op")) {
    args.outPath = getCmdOption(argv, argv + argc, "-op");
  }

  if(cmdOptionExists(argv, argv + argc, "-ip")) {
    args.inputPath = getCmdOption(argv, argv + argc, "-ip");
  }

  if(cmdOptionExists(argv, argv + argc, "-v")) {
    args.verbose = true;
  }

  if(cmdOptionExists(argv, argv + argc, "-t")) {
    args.templateName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Template file Input is required!");
    return;
  }

  if(cmdOptionExists(argv, argv + argc, "-s")) {
    args.scienceName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Science file input is required!");
    return;
  }
}

#endif
