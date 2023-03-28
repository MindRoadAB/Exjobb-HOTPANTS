#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <CL/opencl.hpp>
#include <vector>

#include "argsUtil.h"
#include "fitsUtil.h"
#include "clUtil.h"

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

  readImage(templateImg);

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};
  
  cl::Program::Sources sources;
  string kernel_code = get_kernel_func("conv.cl", "");
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if(program.build({default_device}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
              << "\n";
    exit(1);
  }
  
  cl_long w = templateImg.axis.first;
  cl_long h = templateImg.axis.second;

  double* inputImage = new double[w * h];
  for(size_t i = 0; i < templateImg.data.size(); i++) {
    inputImage[i] = templateImg.data[i];
    // std::cout << inputImage[i] << " ";
  }

  cl::Buffer imgbuf(context, CL_MEM_READ_WRITE, sizeof(double) * w * h);
  cl::Buffer outimgbuf(context, CL_MEM_READ_WRITE, sizeof(double) * w * h);

  // box 5x5
  cl_long kernWidth = 5;
  double a = 1.0 / (double) (kernWidth * kernWidth);
  double convKern[] = {a, a, a, a, a, a, a, a, a, a, a, a, a,
                      a, a, a, a, a, a, a, a, a, a, a, a};
  cl::Buffer kernbuf(context, CL_MEM_READ_WRITE, sizeof(double) * kernWidth * kernWidth);
  
  cl::CommandQueue queue(context, default_device);

  queue.enqueueWriteBuffer(kernbuf, CL_TRUE, 0, sizeof(double) * kernWidth * kernWidth, convKern);
  queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(double) * w * h, inputImage);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long, cl_long> conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h), cl::NullRange};
  conv(eargs, kernbuf, kernWidth, imgbuf, outimgbuf, w, h);

  double* outputImage = new double[w * h];
  cl_int err = queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0, sizeof(double) * w * h,
                          outputImage);

  std::valarray<double> outDat{outputImage, (size_t) (w * h)};
  for(long i = 0; i < w * h; i++) {
    // outDat += outputImage[i];
    if(outputImage == 0) {
      std::cout << "pix nr." << i << " is zero\n";
    }
    // std::cout << outDat[i] << " ";
  }

  Image outImg{args.outName, args.outPath, outDat, templateImg.axis};
  writeImage(outImg);

  return 0;
}