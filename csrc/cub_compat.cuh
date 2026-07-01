#pragma once

namespace vllm::cub_compat {

struct Sum {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a + b;
  }
};

struct Max {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a < b ? b : a;
  }
};

}  // namespace vllm::cub_compat
