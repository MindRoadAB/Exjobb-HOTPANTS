#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

#include "argsUtil.h"
#include "clUtil.h"
#include "fitsUtil.h"

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

  for(double db : templateImg.data) {
    std::cout << db << std::endl;
  }

  readImage(templateImg);

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program::Sources sources;
  string kernel_code = get_kernel_func("simple_conv.cl", "");
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if(program.build({default_device}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
              << "\n";
    exit(1);
  }

  auto [w, h] = templateImg.axis;

  double* inputImage = new double[w * h];
  for(size_t i = 0; i < templateImg.data.size(); i++) {
    inputImage[i] = templateImg.data[i];
  }

  cl::Buffer imgbuf(context, CL_MEM_READ_WRITE, sizeof(double) * w * h);
  cl::Buffer outimgbuf(context, CL_MEM_READ_WRITE, sizeof(double) * w * h);

  cl::CommandQueue queue(context, default_device);

  queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(double) * w * h,
                           inputImage);

  cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer> conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, imgbuf, w, h, outimgbuf);

  double* outputImage = new double[w * h];
  queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0, sizeof(double) * w * h,
                          outputImage);
  long count = 0;
  long count1 = 0;
  for(size_t i = 0; i < templateImg.data.size(); i++) {
    if(outputImage[i] != inputImage[i]) {
      std::cout << "ad: " << outputImage[i] << std::endl;
      count++;
    }
    count1++;
  }
  std::cout << count1 << ' ' << count << std::endl;

  Image outImg{args.outName, args.outPath, templateImg.data, templateImg.axis};
  writeImage(outImg);

  return 0;
}
