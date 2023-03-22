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
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    cl::Context context({default_device});
    cl::Program::Sources sources;
    string kernel_code = get_kernel_func("conv.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    int w, h, max;
    get_ppm_metadata("im1.ppm", &w, &h, &max);
    pixel *image = new pixel[w*h]; 
    read_ppm("im1.ppm", &w, &h, &max, (char*) image);
    cout << (int)image[w*h-1].r << endl;

    cl::Buffer imgbuf(context, CL_MEM_READ_WRITE,sizeof(pixel)*w*h);
    cl::Buffer outimgbuf(context, CL_MEM_READ_WRITE,sizeof(pixel)*w*h);

    cl::Buffer kernbuf(context, CL_MEM_READ_WRITE,sizeof(float)*9);
    // gaussian
    //float a = 1.0f/16.0f;
    //float b = 2.0f/16.0f;
    //float c = 4.0f/16.0f;
    //float convkern[] = {a, b, a, b, c, b, a, b, a};
    
    // box
    //float a = 1.0f/9.0f;
    //float convkern[] = {a, a, a, a, a, a, a, a, a};
    
    // ridge/edge
    //float a = 0.0f;
    //float b = -1.0f;
    //float c = 4.0f;
    //float convkern[] = {a, b, a, b, c, b, a, b, a};
    
    // ridge/edge
    //float a = -1.0f;
    //float b = 8.0f;
    //float convkern[] = {a, a, a, a, b, a, a, a, a};
    
    // sharp
    float a = 0.0f;
    float b = -0.5f;
    float c = 3.0f;
    float convkern[] = {a, b, a, b, c, b, a, b, a};
    

    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

    cl::CommandQueue queue(context,default_device);

    queue.enqueueWriteBuffer(kernbuf,CL_TRUE,0,sizeof(float)*9, convkern);
    queue.enqueueWriteBuffer(imgbuf,CL_TRUE,0,sizeof(pixel)*w*h, image);

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, cl::Buffer> conv{program, "conv"};
    cl::EnqueueArgs eargs{queue,cl::NullRange,cl::NDRange(w*h),cl::NullRange};
    conv(eargs, kernbuf, imgbuf, outimgbuf, w, h, buffer_C);

    pixel *outimg = new pixel[w*h]; 
    queue.enqueueReadBuffer(outimgbuf,CL_TRUE,0,sizeof(pixel)*w*h,outimg);
    int C[10];
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

    for(int i=0;i<10;i++){
        cout<<C[i]<<" ";
    }

    write_ppm("out.ppm", w, h, (char*) outimg);
    write_ppm("in.ppm", w, h, (char*) image);
}
