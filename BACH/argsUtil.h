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
};

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

Arguments getArguments(int argc, char* argv[]) {
  Arguments args;
  if(cmdOptionExists(argv, argv + argc, "-o")) {
    args.outName = getCmdOption(argv, argv + argc, "-o");
  }

  if(cmdOptionExists(argv, argv + argc, "-op")) {
    args.outPath = getCmdOption(argv, argv + argc, "-op");
  }

  if(cmdOptionExists(argv, argv + argc, "-ip")) {
    args.inputPath = getCmdOption(argv, argv + argc, "-ip");
  }

  if(cmdOptionExists(argv, argv + argc, "-t")) {
    args.templateName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Template file Input is required!");
    return args;
  }

  if(cmdOptionExists(argv, argv + argc, "-s")) {
    args.scienceName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Science file input is required!");
    return args;
  }

  return args;
}