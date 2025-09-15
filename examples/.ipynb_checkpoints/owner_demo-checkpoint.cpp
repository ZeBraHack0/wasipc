#include "wasipc/wasipc.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

__global__ void fill_pattern(uint16_t* p, size_t elems, uint16_t base) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elems) p[i] = base + (i % 1024);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: owner_demo <device_id> <handle_file>\n"; return 1;
  }
  int dev = std::stoi(argv[1]);
  std::string path = argv[2];
  cudaSetDevice(dev);

  // 假设一层 FFN 大小为 256 MiB
  const size_t bytes = 256ull << 20;
  void* buf = nullptr;
  cudaMalloc(&buf, bytes);

  // 填充 BF16 张量
  size_t elems = bytes / 2;
  fill_pattern<<<(elems+255)/256, 256>>>((uint16_t*)buf, elems, /*base=*/123);
  cudaDeviceSynchronize();

  // 注册并导出句柄
  wasipc::RegionHandle h{};
  auto st = wasipc::register_region(buf, bytes, &h);
  if (st != cudaSuccess) {
    std::cerr << "register_region failed: " << cudaGetErrorString(st) << "\n"; return 2;
  }

  // 将句柄序列化写到文件
  auto bytes_vec = wasipc::serialize(h);
  std::ofstream ofs(path, std::ios::binary);
  ofs.write((const char*)bytes_vec.data(), bytes_vec.size());
  ofs.close();

  std::cout << "Owner ready. Handle written to " << path << "\n";
  std::cout << "Press ENTER to exit...\n";
  std::cin.get();

  wasipc::deregister_region(buf);
  cudaFree(buf);
  return 0;
}
