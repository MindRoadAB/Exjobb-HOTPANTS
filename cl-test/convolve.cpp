#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <CL/opencl.hpp>
#include <cstdlib>
#include "ppmio.h"

#include "clUtil.h"

using namespace std;

typedef struct _pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;} pixel;

int main() {
    cl::Device default_device{get_default_device()};
    cl::Context context({default_device});

    cl::Program::Sources sources;
    string kernel_code = get_kernel_func("conv_old.cl", "");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context, sources);
    if(program.build({default_device}) != CL_SUCCESS) {
      std::cout << " Error building: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
                << "\n";
      exit(1);
    }

    int w, h, max;
    get_ppm_metadata("im1.ppm", &w, &h, &max);
    pixel *image = new pixel[w * h];
    read_ppm("im1.ppm", &w, &h, &max, (char *)image);
    cout << (int)image[w * h - 1].r << endl;

    cl::Buffer imgbuf(context, CL_MEM_READ_WRITE, sizeof(pixel) * w * h);
    cl::Buffer outimgbuf(context, CL_MEM_READ_WRITE, sizeof(pixel) * w * h);

    // gaussian
    // int kernWidth = 3;
    // float a = 1.0f/16.0f;
    // float b = 2.0f/16.0f;
    // float c = 4.0f/16.0f;
    // float convkern[] = {a, b, a, b, c, b, a, b, a};

    // box
    // int kernWidth = 3;
    // float a = 1.0f/9.0f;
    // float convkern[] = {a, a, a, a, a, a, a, a, a};

    // box 2x2
    // int kernWidth = 2;
    // float a = 1.0f / 4.0f;
    // float convkern[] = {a, a, a, a};

    // box 5x5
    // int kernWidth = 5;
    // float a = 1.0f / 25.0f;
    // float convkern[] = {a, a, a, a, a, a, a, a, a, a, a, a, a,
    //                     a, a, a, a, a, a, a, a, a, a, a, a};

    // ridge/edge
    // int kernWidth = 3;
    // float a = 0.0f;
    // float b = -1.0f;
    // float c = 4.0f;
    // float convkern[] = {a, b, a, b, c, b, a, b, a};

    // // ridge/edge
    // int kernWidth = 3;
    // float a = -1.0f;
    // float b = 8.0f;
    // float convkern[] = {a, a, a, a, b, a, a, a, a};

    // sharp
    int kernWidth = 3;
    float a = 0.0f;
    float b = -0.5f;
    float c = 3.0f;
    float convkern[] = {a, b, a, b, c, b, a, b, a};

    cl::Buffer kernbuf(context, CL_MEM_READ_WRITE,
                       sizeof(float) * kernWidth * kernWidth);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE,
                        sizeof(int) * kernWidth * kernWidth);

    cl::CommandQueue queue(context, default_device);

    queue.enqueueWriteBuffer(kernbuf, CL_TRUE, 0,
                             sizeof(float) * kernWidth * kernWidth, convkern);
    queue.enqueueWriteBuffer(imgbuf, CL_TRUE, 0, sizeof(pixel) * w * h, image);

    cl::KernelFunctor<cl::Buffer, int, cl::Buffer, cl::Buffer, int, int,
                      cl::Buffer>
        conv{program, "conv"};
    cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                          cl::NullRange};
    conv(eargs, kernbuf, kernWidth, imgbuf, outimgbuf, w, h, buffer_C);

    pixel *outimg = new pixel[w * h];
    queue.enqueueReadBuffer(outimgbuf, CL_TRUE, 0, sizeof(pixel) * w * h,
                            outimg);
    int C[25];
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0,
                            sizeof(int) * kernWidth * kernWidth, C);

    for(int i = 0; i < kernWidth * kernWidth; i++) {
      cout << C[i] << " ";
    }

    write_ppm("out.ppm", w, h, (char *)outimg);
    write_ppm("in.ppm", w, h, (char *)image);
}
