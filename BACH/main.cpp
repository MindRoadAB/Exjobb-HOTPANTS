#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

#include "utils/argsUtil.h"
#include "utils/bachUtil.h"
#include "utils/clUtil.h"
#include "utils/fitsUtil.h"

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

  if(args.verbose)
    std::cout << "template image name: " << args.templateName
              << ", science image name: " << args.scienceName << std::endl;

  cl_int err{};

  err = readImage(templateImg);
  checkError(err);
  // TODO: Maybe do maskInput in parallel?
  maskInput(templateImg);
  err = readImage(scienceImg);
  checkError(err);
  maskInput(scienceImg);

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program program =
      load_build_programs(context, default_device, "conv.cl", "sub.cl");

  auto [w, h] = templateImg.axis;
  if(w != scienceImg.axis.first || h != scienceImg.axis.second) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;

    exit(1);
  }

  /* ===== SSS ===== */

  std::vector<Stamp> templateStamps{};
  createStamps(templateImg, templateStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << templateImg.name << std::endl;
  }

  std::vector<Stamp> sciStamps{};
  createStamps(scienceImg, sciStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << scienceImg.name << std::endl;
  }

  /* == Check Template Stamps  ==*/
  int numTemplSStamps = identifySStamps(templateStamps, templateImg);
  if(cl_double(numTemplSStamps) / templateStamps.size() < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in " << templateImg.name
                << " trying again with lower thresholds..." << std::endl;
    args.threshLow *= 0.5;
    numTemplSStamps = identifySStamps(templateStamps, templateImg);
    args.threshLow /= 0.5;
  }
  if(args.verbose) {
    std::cout << "Substamps found in " << templateImg.name << std::endl;
  }

  /* == Check Science Stamps  ==*/
  int numSciSStamps = identifySStamps(sciStamps, scienceImg);
  if(cl_double(numSciSStamps) / sciStamps.size() < 0.1) {
    if(args.verbose) {
      std::cout << "Not enough substamps found in " << scienceImg.name
                << " trying again with lower thresholds..." << std::endl;
    }
    args.threshLow *= 0.5;
    numSciSStamps = identifySStamps(sciStamps, scienceImg);
    args.threshLow /= 0.5;
  }
  if(args.verbose) {
    std::cout << "Substamps found in " << scienceImg.name << std::endl;
  }

  if(numTemplSStamps == 0 && numSciSStamps == 0) {
    std::cout << "No substamps found" << std::endl;
    exit(1);
  }

  /* ===== CMV ===== */

  if(args.verbose) std::cout << "Calculating matrix variables..." << std::endl;
  Kernel convolutionKernel{};
  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, convolutionKernel);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, convolutionKernel);
  }

  /* ===== CD ===== */

  std::cout << "Choosing Convolution Direction" << std::endl;
  std::vector<int> index(templateStamps[1].Q.size());
  double d;
  std::vector<std::vector<cl_double>> input = {{0.0, 0.0, 0.0, 0.0},
                                               {0.0, 3.123, 5.1, 23.5},
                                               {0.0, 22.5, 50.5, 30.0},
                                               {0.0, 33.0, 34.23, 20.0}};
  std::vector<cl_double> output = {0.0, 2.0, 3.0, 4.0};

  // ludcmp(input, 3, index, d);
  // lubksb(input, 3, index, output);

  std::vector<cl_double> testVec(args.nPSF + 2, 0.0);
  std::vector<std::vector<cl_double>> testMat(
      args.nPSF + 2, std::vector<cl_double>(args.nPSF + 2, 0.0));

  for(int i = 1; i <= args.nPSF + 1; i++) {
    testVec[i] = templateStamps[0].B[i];
    for(int j = 1; j <= i; j++) {
      testMat[i][j] = templateStamps[0].Q[i][j];
      testMat[j][i] = testMat[i][j];
    }
  }

  ludcmp(testMat, args.nPSF + 1, index, d);
  lubksb(testMat, args.nPSF + 1, index, testVec);
  for(auto e : testVec) {
    std::cout << e << std::endl;
  }

  /* ===== Conv ===== */
  std::cout << "Doing Convolution" << std::endl;

  cl::Buffer imgbuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer outimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);

  // box 5x5
  cl_long kernWidth = 5;
  cl_double a = 1.0 / (cl_double)(kernWidth * kernWidth);
  cl_double convKern[] = {a, a, a, a, a, a, a, a, a, a, a, a, a,
                          a, a, a, a, a, a, a, a, a, a, a, a};

  cl::Buffer kernbuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * kernWidth * kernWidth);

  cl::CommandQueue queue(context, default_device);

  err = queue.enqueueWriteBuffer(
      kernbuf, CL_TRUE, 0, sizeof(cl_double) * kernWidth * kernWidth, convKern);
  checkError(err);

  err = queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &templateImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, kernbuf, kernWidth, imgbuf, outimgbuf, w, h);

  Image outImg{args.outName, templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &outImg);
  checkError(err);

  err = writeImage(outImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> sub{program, "sub"};
  sub(eargs, outimgbuf, imgbuf, diffimgbuf);

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(diffimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &diffImg);
  checkError(err);

  err = writeImage(diffImg);
  checkError(err);

  return 0;
}
