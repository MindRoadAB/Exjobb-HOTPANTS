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
  // TODO: Maybe do maskInput in parallel?
  maskInput(templateImg);
  err = readImage(scienceImg);
  checkError(err);
  maskInput(scienceImg);

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

  cl_double templateMerit = testFit(templateStamps, templateImg);
  cl_double scienceMerit = testFit(sciStamps, scienceImg);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;
  if(scienceMerit < templateMerit) {
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

  std::vector<std::vector<cl_double>> convKernels{};
  int xSteps =
      std::ceil((templateImg.axis.first) / cl_double(args.fKernelWidth));
  int ySteps =
      std::ceil((templateImg.axis.second) / cl_double(args.fKernelWidth));
  for(int x = 0; x < xSteps; x++) {
    int imgX = x * xSteps + args.hKernelWidth;
    for(int y = 0; y < ySteps; y++) {
      int imgY = y * ySteps + args.hKernelWidth;
      makeKernel(convolutionKernel, templateImg.axis, imgX, imgY);
      convKernels.push_back(convolutionKernel.currKernel);
    }
  }

  cl::Buffer imgbuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer outimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffimgbuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer kernbuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size() *
                         args.fKernelWidth * args.fKernelWidth);
  // box 5x5
  // cl_long kernWidth = 5;
  // cl_double a = 1.0 / (cl_double)(kernWidth * kernWidth);
  // cl_double convKern[] = {a, a, a, a, a, a, a, a, a, a, a, a, a,
  //                         a, a, a, a, a, a, a, a, a, a, a, a};
  // cl::Buffer kernbuf(context, CL_MEM_READ_ONLY,
  //                    sizeof(cl_double) * kernWidth * kernWidth);

  cl::CommandQueue queue(context, default_device);

  err = queue.enqueueWriteBuffer(kernbuf, CL_TRUE, 0,
                                 sizeof(cl_double) * convKernels.size() *
                                     args.fKernelWidth * args.fKernelWidth,
                                 &convKernels[0][0]);
  checkError(err);

  err = queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &templateImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  std::cout << "Debug 1" << std::endl;
  conv(eargs, kernbuf, args.fKernelWidth, imgbuf, outimgbuf, w, h);
  std::cout << "Debug 2" << std::endl;

  Image outImg{args.outName, templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &outImg);
  checkError(err);

  std::cout << std::endl;

  err = writeImage(outImg);
  checkError(err);

  std::cout << "Subtracting images..." << std::endl;

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> sub{program, "sub"};
  sub(eargs, outimgbuf, imgbuf, diffimgbuf);

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(diffimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &diffImg);
  checkError(err);

  std::cout << std::endl;

  std::cout << "Writing output..." << std::endl;

  err = writeImage(diffImg);
  checkError(err);

  std::cout << std::endl;

  std::cout << "BACH finished." << std::endl;
  return 0;
}
