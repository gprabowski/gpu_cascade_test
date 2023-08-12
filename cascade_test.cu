#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "test.cuh"

int main() {
  // 1. Read the file
  std::ifstream input("test.json");
  if (!input.good()) {
    std::cerr << "File doesn't exist" << std::endl;
    exit(-1);
  }
  std::stringstream ss;
  ss << input.rdbuf();
  // 2. Filter
  auto res = ss.str();
  testing::test(res);

  return 0;
}
