#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "gpu_filter.cuh"
#include <cooperative_groups.h>

namespace testing {

namespace cg = cooperative_groups;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

struct is_newline {
  __host__ __device__ bool operator()(const char *a) { return *a == '\n'; }
};

// currently ASCII assumption
size_t test(std::string &lines) {
  const auto byte_count = lines.size();
  const auto len = lines.size() + 1;
  const char *h_text = lines.c_str();
  char *d_text;
  char *d_is_valid;
  char **d_addresses;

  cudaMalloc(&d_text, len);
  cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice);

  const auto json_count =
      thrust::count(thrust::device, d_text, d_text + len, '\n');

  cudaMalloc(&d_addresses, json_count * sizeof(char *));
  cudaMalloc(&d_is_valid, json_count * sizeof(char));

  thrust::copy_if(thrust::device, thrust::make_counting_iterator(d_text),
                  thrust::make_counting_iterator(d_text + len), d_addresses,
                  is_newline());

  std::cout << "JSON COUNT: " << json_count << std::endl;

  // Run warp filter
  filter::gpu_filter()(d_text, static_cast<size_t>(json_count), d_addresses,
                       d_is_valid);

  const auto correct_count =
      thrust::reduce(thrust::device, d_is_valid, d_is_valid + json_count, 0);
  std::cout << " VALID: " << correct_count << std::endl;
  cudaFree(d_addresses);
  cudaFree(d_text);
  return correct_count;
}

} // namespace testing
