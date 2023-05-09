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

  std::cout << "\nReading in images..." << std::endl;

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

  std::cout << "\nSetting up openCL..." << std::endl;

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program program =
      load_build_programs(context, default_device, "conv.cl", "sub.cl");

  /* ===== SSS ===== */

  std::cout << "\nCreating stamps..." << std::endl;

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

  /* ===== CMV ===== */

  std::cout << "\nCalculating matrix variables..." << std::endl;

  Kernel convolutionKernel{};
  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, convolutionKernel);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, convolutionKernel);
  }

  /* ===== CD ===== */

  std::cout << "\nChoosing convolution direction..." << std::endl;

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

  /* ===== KSC ===== */

  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg);

  /* ===== Conv ===== */

  std::cout << "\nConvolving..." << std::endl;

  // Convolution kernels generated beforehand since we only need on per
  // kernelsize.
  std::vector<cl_double> convKernels{};
  int xSteps = std::ceil((templateImg.axis.first) / double(args.fKernelWidth));
  int ySteps = std::ceil((templateImg.axis.second) / double(args.fKernelWidth));
  for(int yStep = 0; yStep < ySteps; yStep++) {
    for(int xStep = 0; xStep < xSteps; xStep++) {
      makeKernel(
          convolutionKernel, templateImg.axis,
          xStep * args.fKernelWidth + args.hKernelWidth + args.hKernelWidth,
          yStep * args.fKernelWidth + args.hKernelWidth + args.hKernelWidth);
      convKernels.insert(convKernels.end(),
                         convolutionKernel.currKernel.begin(),
                         convolutionKernel.currKernel.end());
    }
  }

  // Used to normalize the result since the kernel sum is not always 1.
  double kernSum =
      makeKernel(convolutionKernel, templateImg.axis,
                 templateImg.axis.first / 2, templateImg.axis.second / 2);
  cl_double invKernSum = 1.0 / kernSum;

  if(args.verbose) {
    std::cout << "Sum of kernel at (" << templateImg.axis.first / 2 << ","
              << templateImg.axis.second / 2 << "): " << kernSum << std::endl;
  }

  // Declare all the buffers which will be need in opencl operations.
  cl::Buffer tImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer sImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer convImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer kernBuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());
  cl::Buffer outImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);

  cl::CommandQueue queue(context, default_device);

  // Write necessary data for convolution
  err = queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * convKernels.size(),
                                 &convKernels[0]);
  checkError(err);
  err = queue.enqueueWriteBuffer(tImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &templateImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, kernBuf, args.fKernelWidth, tImgBuf, outImgBuf, w, h);

  // Read data from convolution
  Image outImg{args.outName, templateImg.axis, args.outPath};

  err = queue.enqueueReadBuffer(outImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &outImg);
  checkError(err);

  // Add background and scale by kernel sum for output of convoluted image.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis);
      outImg.data[x + y * w] *= invKernSum;
    }
  }

  err = writeImage(outImg);
  checkError(err);

  // Revert scale by sum kernel for use in subtraction.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] *= kernSum;
    }
  }

  std::cout << "\nSubtracting images..." << std::endl;

  // Write necessary data for subtraction
  err = queue.enqueueWriteBuffer(convImgBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * w * h, &outImg);
  err = queue.enqueueWriteBuffer(sImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &scienceImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_double> sub{program,
                                                                       "sub"};
  sub(eargs, sImgBuf, convImgBuf, diffImgBuf, invKernSum);

  // Read data from subtraction
  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(diffImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &diffImg);
  checkError(err);

  std::cout << "\nWriting output..." << std::endl;

  err = writeImage(diffImg);
  checkError(err);

  std::cout << "\nBACH finished." << std::endl;
  return 0;
}
