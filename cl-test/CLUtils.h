#include <chrono>
#include <fstream>
#include <iostream>

std::string const DEFAULT_PATH = "./";
std::string get_kernel_func(std::string &&file_name,
                            std::string &&path = "NULL") {
  if (path == "NULL") {
    path = DEFAULT_PATH;
  }
  std::ifstream t(path + file_name);
  std::string tmp{std::istreambuf_iterator<char>{t},
                  std::istreambuf_iterator<char>{}};

  return tmp;
}

auto get_time() -> decltype(std::chrono::high_resolution_clock::now()) {
  auto tmp{std::chrono::high_resolution_clock::now()};
  return tmp;
}

// void print_time(std::ostream &os, auto start, auto stop) {
//   os << "Time in ms: "
//      << std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
//             .count()
//      << std::endl;
// }
