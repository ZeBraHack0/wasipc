#pragma once
#include <cstdint>
#include <vector>
#include <cstring>

#include <cuda_runtime_api.h>

namespace wasipc {

// 确保与对端进程可序列化/反序列化的固定布局
#pragma pack(push, 1)
struct RegionHandle {
  cudaIpcMemHandle_t mem_handle; // CUDA IPC 句柄（二进制可直接传输）
  int32_t owner_device;          // 拥有该区域的 device ordinal
  uint32_t reserved0{0};
  uint64_t bytes{0};             // 区域总字节
  uint64_t abi_version{1};       // 兼容版本
  uint64_t tag{0};               // 可选：layer_id 等业务标记
};
#pragma pack(pop)

struct RegionInfo {
  void*   remote_ptr{nullptr}; // 远端显存 UVA 指针（本进程可用）
  int32_t owner_device{-1};
  uint64_t bytes{0};
};

// -------- Owner 侧 API --------

// 将一段由 cudaMalloc 得到的连续显存暴露给其他进程
cudaError_t register_region(void* device_ptr, uint64_t bytes, RegionHandle* out);

// Owner 侧“取消导出”（便于做生命周期自检；CUDA 无显式注销，函数本身是幂等 no-op）
cudaError_t deregister_region(void* device_ptr);

// -------- Requester 侧 API --------

// 打开远端区域，获得可解引用的远端 UVA 指针（本进程）
cudaError_t open_region(const RegionHandle& h, RegionInfo* out,
                        unsigned int flags = cudaIpcMemLazyEnablePeerAccess);

// 关闭远端区域（释放本进程的映射）
cudaError_t close_region(RegionInfo* info);

// 异步单次读取：把远端 [src_off, src_off+bytes) 复制到本地 dst_device（DeviceToDevice）
cudaError_t read_async(const RegionInfo& src, void* dst_device,
                       uint64_t bytes, uint64_t src_off,
                       cudaStream_t stream);

// 批量异步读取（减少 API 调用开销）
struct ReadDesc { void* dst; uint64_t bytes, src_off; cudaStream_t stream; };
cudaError_t readv_async(const RegionInfo& src, const ReadDesc* descs, int n);

// -------- 辅助工具 --------
bool can_access_peer(int requester_dev, int owner_dev);
cudaError_t enable_peer_access(int requester_dev, int owner_dev);

// 句柄序列化/反序列化（跨进程传递）
inline std::vector<uint8_t> serialize(const RegionHandle& h) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&h);
  return std::vector<uint8_t>(p, p + sizeof(RegionHandle));
}
inline bool deserialize(const void* data, size_t n, RegionHandle* out) {
  if (n != sizeof(RegionHandle) || out == nullptr) return false;
  std::memcpy(out, data, sizeof(RegionHandle)); return true;
}

} // namespace wasipc
