#include <CL/opencl.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

inline cl::Device get_default_device() {
  // get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if(all_platforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    exit(1);
  }
  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  // get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()
            << "\n";

  return default_device;
}

std::string const DEFAULT_PATH = "cl_kern/";
inline std::string get_kernel_func(std::string &&file_name,
                                   std::string &&path = "NULL") {
  if(path == "NULL") {
    path = DEFAULT_PATH;
  }
  std::ifstream t(path + file_name);
  std::string tmp{std::istreambuf_iterator<char>{t},
                  std::istreambuf_iterator<char>{}};

  return tmp;
}

inline auto get_time() -> decltype(std::chrono::high_resolution_clock::now()) {
  auto tmp{std::chrono::high_resolution_clock::now()};
  return tmp;
}

using timePoint = std::chrono::high_resolution_clock::time_point;

inline void print_time(std::ostream &os, timePoint start, timePoint stop) {
  os << "Time in ms: "
     << std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count()
     << std::endl;
}
