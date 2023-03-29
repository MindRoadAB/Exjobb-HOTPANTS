#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

#include "argsUtil.h"
#include "clUtil.h"
#include "fitsUtil.h"

void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

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
  cl_int err{};

  err = readImage(templateImg);
  checkError(err);

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program::Sources sources;
  std::string convCode = get_kernel_func("conv.cl");
  sources.push_back({convCode.c_str(), convCode.length()});
  std::string subCode = get_kernel_func("sub.cl");
  sources.push_back({subCode.c_str(), subCode.length()});

  cl::Program program(context, sources);
  if(program.build({default_device}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
              << "\n";
    exit(1);
  }

  cl_long w = templateImg.axis.first;
  cl_long h = templateImg.axis.second;

  cl_double* inputImage = new cl_double[w * h];
  for(cl_uint i = 0; i < templateImg.data.size(); i++) {
    inputImage[i] = templateImg.data[i];
  }

  cl::Buffer imgbuf(context, CL_MEM_READ_WRITE, sizeof(cl_double) * w * h);
  cl::Buffer outimgbuf(context, CL_MEM_READ_WRITE, sizeof(cl_double) * w * h);
  cl::Buffer diffimgbuf(context, CL_MEM_READ_WRITE, sizeof(cl_double) * w * h);

  // box 5x5
  cl_long kernWidth = 5;
  cl_double a = 1.0 / (cl_double)(kernWidth * kernWidth);
  cl_double convKern[] = {a, a, a, a, a, a, a, a, a, a, a, a, a,
                          a, a, a, a, a, a, a, a, a, a, a, a};

  cl::Buffer kernbuf(context, CL_MEM_READ_WRITE,
                     sizeof(cl_double) * kernWidth * kernWidth);

  cl::CommandQueue queue(context, default_device);

  err = queue.enqueueWriteBuffer(
      kernbuf, CL_TRUE, 0, sizeof(cl_double) * kernWidth * kernWidth, convKern);
  checkError(err);
  err = queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 inputImage);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, kernbuf, kernWidth, imgbuf, outimgbuf, w, h);

  cl_double* outputImage = new cl_double[w * h];
  err = queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, outputImage);
  checkError(err);

  std::valarray<cl_double> outDat{outputImage, (size_t)(w * h)};
  Image outImg{args.outName, args.outPath, outDat, templateImg.axis};
  err = writeImage(outImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> sub{program, "sub"};
  sub(eargs, outimgbuf, imgbuf, diffimgbuf);

  cl_double* diffImage = new cl_double[w * h];
  err = queue.enqueueReadBuffer(diffimgbuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, diffImage);
  checkError(err);

  std::valarray<cl_double> diffDat{diffImage, (size_t)(w * h)};
  Image diffImg{"sub.fits", args.outPath, diffDat, templateImg.axis};
  err = writeImage(diffImg);
  checkError(err);

  return 0;
}
