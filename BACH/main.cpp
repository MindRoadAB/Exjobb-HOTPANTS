#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <iterator>
#include <vector>

#include "utils/argsUtil.h"
#include "utils/bachUtil.h"
#include "utils/clUtil.h"
#include "utils/fitsUtil.h"

int main(int argc, char* argv[]) {
  CCfits::FITS::setVerboseMode(true);

  try {
    std::cout << "Reading in arguments..." << std::endl;
    getArguments(argc, argv);
  } catch(const std::invalid_argument& err) {
    std::cout << err.what() << '\n';
    return 1;
  }

  std::cout << std::endl;

  std::cout << "Reading in images..." << std::endl;

  Image templateImg{args.templateName};
  Image scienceImg{args.scienceName};

  if(args.verbose)
    std::cout << "template image name: " << args.templateName
              << ", science image name: " << args.scienceName << std::endl;

  cl_int err{};

  err = readImage(templateImg);
  checkError(err);
  err = readImage(scienceImg);
  checkError(err);

  maskInput(templateImg, scienceImg);

  auto [w, h] = templateImg.axis;
  if(w != scienceImg.axis.first || h != scienceImg.axis.second) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;

    exit(1);
  }

  std::cout << std::endl;

  std::cout << "Setting up openCL..." << std::endl;

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program program =
      load_build_programs(context, default_device, "conv.cl", "sub.cl");

  std::cout << std::endl;

  /* ===== SSS ===== */

  std::cout << "Creating stamps..." << std::endl;

  args.fStampWidth = std::min(int(templateImg.axis.first / args.stampsx),
                              int(templateImg.axis.second / args.stampsy));
  args.fStampWidth -= args.fKernelWidth;
  args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

  if(args.fStampWidth < args.fSStampWidth) {
    args.fStampWidth = args.fSStampWidth + args.fKernelWidth;
    args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

    args.stampsx = int(templateImg.axis.first / args.fStampWidth);
    args.stampsy = int(templateImg.axis.second / args.fStampWidth);

    if(args.verbose)
      std::cout << "Too many stamps requested, using " << args.stampsx << "x"
                << args.stampsy << " stamps instead." << std::endl;
  }

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
  if(double(numTemplSStamps) / templateStamps.size() < 0.1) {
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
  if(double(numSciSStamps) / sciStamps.size() < 0.1) {
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

  std::cout << std::endl;

  /* ===== CMV ===== */

  std::cout << "Calculating matrix variables..." << std::endl;

  Kernel convolutionKernel{};
  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, convolutionKernel);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, convolutionKernel);
  }

  std::cout << std::endl;

  /* ===== CD ===== */

  std::cout << "Choosing convolution direction..." << std::endl;

  double templateMerit = testFit(templateStamps, templateImg, scienceImg);
  double scienceMerit = testFit(sciStamps, scienceImg, templateImg);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;
  if(scienceMerit <= templateMerit) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
  }
  if(args.verbose)
    std::cout << templateImg.name << " chosen to be convolved." << std::endl;

  std::cout << std::endl;

  /* ===== KSC ===== */

  std::cout << "Fitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg);

  std::cout << std::endl;

  /* ===== Conv ===== */

  std::cout << "Convolving..." << std::endl;

  std::vector<cl_double> convKernels{};
  int xSteps = std::ceil((templateImg.axis.first) / double(args.fKernelWidth));
  int ySteps = std::ceil((templateImg.axis.second) / double(args.fKernelWidth));
  for(int y = 0; y < ySteps; y++) {
    int imgY = y * args.fKernelWidth + args.hKernelWidth;
    for(int x = 0; x < xSteps; x++) {
      int imgX = x * args.fKernelWidth + args.hKernelWidth;
      makeKernel(convolutionKernel, templateImg.axis, imgX + args.hKernelWidth,
                 imgY + args.hKernelWidth);
      convKernels.insert(convKernels.end(),
                         convolutionKernel.currKernel.begin(),
                         convolutionKernel.currKernel.end());
    }
  }

  double kernSum =
      makeKernel(convolutionKernel, templateImg.axis,
                 templateImg.axis.first / 2, templateImg.axis.second / 2);
  cl_double invKernSum = 1.0 / kernSum;
  std::cout << "inv kernsum is " << invKernSum << std::endl;
  std::cout << "kernsum is " << kernSum << std::endl;

  cl::Buffer timgbuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer simgbuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer convimgbuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer kernbuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());

  cl::Buffer outimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);

  cl::CommandQueue queue(context, default_device);

  err = queue.enqueueWriteBuffer(kernbuf, CL_TRUE, 0,
                                 sizeof(cl_double) * convKernels.size(),
                                 &convKernels[0]);
  checkError(err);

  err = queue.enqueueWriteBuffer(
      timgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
      &std::vector<cl_double>(templateImg.data.begin(),
                              templateImg.data.end())[0]);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, kernbuf, args.fKernelWidth, timgbuf, outimgbuf, w, h);

  Image outImg{args.outName, templateImg.axis, args.outPath};
  std::vector<cl_double> tmpOut(outImg.size());
  err = queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &tmpOut[0]);
  checkError(err);

  outImg.data = tmpOut;

  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis);
      outImg.data[x + y * w] *= invKernSum;
    }
  }

  std::cout << std::endl;

  err = writeImage(outImg);
  checkError(err);

  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] *= kernSum;
    }
  }

  std::cout << "Subtracting images..." << std::endl;

  err = queue.enqueueWriteBuffer(
      convimgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
      &std::vector<cl_double>(outImg.data.begin(), outImg.data.end())[0]);
  checkError(err);

  err = queue.enqueueWriteBuffer(
      simgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
      &std::vector<cl_double>(scienceImg.data.begin(),
                              scienceImg.data.end())[0]);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_double> sub{program,
                                                                       "sub"};
  sub(eargs, simgbuf, convimgbuf, diffimgbuf, invKernSum);

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(diffimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &tmpOut[0]);
  checkError(err);

  diffImg.data = tmpOut;

  std::cout << std::endl;

  std::cout << "Writing output..." << std::endl;

  err = writeImage(diffImg);
  checkError(err);

  std::cout << std::endl;

  Image kernelImg{
      "kern.fits",
      std::make_pair(long(args.fKernelWidth), long(args.fKernelWidth)),
      args.outPath};
  std::vector<double> kernel{
      std::next(convKernels.begin(),
                args.fKernelWidth * args.fKernelWidth * 500),
      std::next(convKernels.begin(),
                args.fKernelWidth * args.fKernelWidth * 501)};
  kernelImg.data = kernel;
  writeImage(kernelImg);

  std::cout << "BACH finished." << std::endl;
  return 0;
}
