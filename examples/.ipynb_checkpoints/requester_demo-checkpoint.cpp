#include "wasipc/wasipc.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

__global__ void checksum_u16(const uint16_t* p, size_t elems, unsigned long long* out) {
  __shared__ unsigned long long s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long v = 0;
  if (i < elems) v = p[i];
  atomicAdd(&s, v);
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) *out = s;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: requester_demo <device_id> <owner_device_id> <handle_file>\n"; return 1;
  }
  int dev = std::stoi(argv[1]);
  int owner_dev = std::stoi(argv[2]);
  std::string path = argv[3];

  cudaSetDevice(dev);
  if (!wasipc::can_access_peer(dev, owner_dev)) {
    std::cerr << "P2P not available between dev " << dev << " and owner " << owner_dev << "\n";
    return 2;
  }
  wasipc::enable_peer_access(dev, owner_dev);

  // 读回句柄
  std::ifstream ifs(path, std::ios::binary);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(ifs)), {});
  ifs.close();
  wasipc::RegionHandle h{};
  if (!wasipc::deserialize(buf.data(), buf.size(), &h)) {
    std::cerr << "deserialize failed\n"; return 3;
  }

  // 打开远端区域
  wasipc::RegionInfo info{};
  auto st = wasipc::open_region(h, &info);
  if (st != cudaSuccess) {
    std::cerr << "open_region failed: " << cudaGetErrorString(st) << "\n"; return 4;
  }

  // 分页异步读取 + 校验
  const size_t page = 16ull << 20;
  void* local = nullptr;
  cudaMalloc(&local, page);
  cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

  // 拿第一页做校验演示
  st = wasipc::read_async(info, local, page, /*src_off*/0, s);
  cudaStreamSynchronize(s);
  unsigned long long* dsum; cudaMalloc(&dsum, sizeof(unsigned long long));
  checksum_u16<<<(page/2+255)/256,256>>>((uint16_t*)local, page/2, dsum);
  unsigned long long hsum=0; cudaMemcpy(&hsum, dsum, sizeof(hsum), cudaMemcpyDeviceToHost);
  std::cout << "Checksum(first 16MiB) = " << hsum << "\n";

  cudaFree(dsum);
  cudaStreamDestroy(s);
  cudaFree(local);
  wasipc::close_region(&info);
  return 0;
}
