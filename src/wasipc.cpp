#include "wasipc/wasipc.h"
#include <cuda.h>
#include <cstring>
#include <unordered_set>
#include <mutex>

namespace wasipc {

static std::mutex g_mu;
static std::unordered_set<void*> g_exported; // 仅用于简单生命周期自检

static inline cudaError_t to_cuda(cudaError_t e) { return e; }

bool can_access_peer(int requester_dev, int owner_dev) {
  if (requester_dev == owner_dev) return true;
  int can = 0;
  auto st = cudaDeviceCanAccessPeer(&can, requester_dev, owner_dev);
  return (st == cudaSuccess) && (can != 0);
}

cudaError_t enable_peer_access(int requester_dev, int owner_dev) {
  if (requester_dev == owner_dev) return cudaSuccess;
  int can = 0;
  cudaError_t st = cudaDeviceCanAccessPeer(&can, requester_dev, owner_dev);
  if (st != cudaSuccess) return st;
  if (!can) return cudaErrorInvalidDevice; // 不可达，避免 host-bounce
  int prev = -1;
  cudaGetDevice(&prev);
  if (prev != requester_dev) cudaSetDevice(requester_dev);
  // 已启用则返回 cudaErrorPeerAccessAlreadyEnabled，可忽略
  cudaError_t rc = cudaDeviceEnablePeerAccess(owner_dev, 0);
  if (prev != requester_dev) cudaSetDevice(prev);
  if (rc == cudaErrorPeerAccessAlreadyEnabled) return cudaSuccess;
  return rc;
}

cudaError_t register_region(void* device_ptr, uint64_t bytes, RegionHandle* out) {
  if (!device_ptr || bytes == 0 || !out) return cudaErrorInvalidValue;
  int dev = -1;
  cudaPointerAttributes attr{};
  auto st = cudaPointerGetAttributes(&attr, device_ptr);
#if CUDART_VERSION >= 11000
  if (st != cudaSuccess || attr.type != cudaMemoryTypeDevice) return cudaErrorInvalidDevicePointer;
#else
  if (st != cudaSuccess || attr.memoryType != cudaMemoryTypeDevice) return cudaErrorInvalidDevicePointer;
#endif
  cudaGetDevice(&dev);
  // 导出 IPC 句柄
  cudaIpcMemHandle_t h{};
  st = cudaIpcGetMemHandle(&h, device_ptr);
  if (st != cudaSuccess) return st;
  RegionHandle rh{};
  rh.mem_handle  = h;
  rh.owner_device= dev;
  rh.bytes       = bytes;
  rh.abi_version = 1;
  *out = rh;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_exported.insert(device_ptr);
  }
  return cudaSuccess;
}

cudaError_t deregister_region(void* device_ptr) {
  // CUDA IPC 无显式注销；这里仅做自检/标记
  if (!device_ptr) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lk(g_mu);
  g_exported.erase(device_ptr);
  return cudaSuccess;
}

cudaError_t open_region(const RegionHandle& h, RegionInfo* out,
                        unsigned int flags) {
  if (!out || h.bytes == 0) return cudaErrorInvalidValue;
  int cur=-1; cudaGetDevice(&cur);
  if (!can_access_peer(cur, h.owner_device) && cur != h.owner_device)
    return cudaErrorInvalidDevice; // 禁止退化到 host-bounce
  void* remote_ptr = nullptr;
  auto st = cudaIpcOpenMemHandle(&remote_ptr, h.mem_handle, flags);
  if (st != cudaSuccess) return st;
  out->remote_ptr   = remote_ptr;
  out->owner_device = h.owner_device;
  out->bytes        = h.bytes;
  return cudaSuccess;
}

cudaError_t close_region(RegionInfo* info) {
  if (!info || !info->remote_ptr) return cudaErrorInvalidValue;
  auto st = cudaIpcCloseMemHandle(info->remote_ptr);
  if (st == cudaSuccess) {
    info->remote_ptr = nullptr; info->bytes = 0; info->owner_device = -1;
  }
  return st;
}

static inline bool range_dst_ok(const RegionInfo& dst, uint64_t off, uint64_t n) {
  if (off > dst.bytes) return false;
  if (n > dst.bytes) return false;
  if (off + n > dst.bytes) return false;
  return true;
}

cudaError_t write_async(const void* src_device,
                        const RegionInfo& dst,
                        uint64_t bytes, uint64_t dst_off,
                        cudaStream_t stream) {
  if (!src_device || !dst.remote_ptr || bytes == 0) return cudaErrorInvalidValue;
  if (!range_dst_ok(dst, dst_off, bytes)) return cudaErrorInvalidValue;
  const char* src = static_cast<const char*>(src_device);
  char*       dp  = static_cast<char*>(dst.remote_ptr);
  return cudaMemcpyAsync(dp + dst_off, src, bytes,
                         cudaMemcpyDeviceToDevice, stream);
}

cudaError_t writev_async(const RegionInfo& dst,
                         const WriteDesc* descs, int n) {
  if (!dst.remote_ptr || !descs || n <= 0) return cudaErrorInvalidValue;
  for (int i = 0; i < n; ++i) {
    auto st = write_async(descs[i].src, dst,
                          descs[i].bytes, descs[i].dst_off,
                          descs[i].stream);
    if (st != cudaSuccess) return st;
  }
  return cudaSuccess;
}

// ---- 事件 IPC ----
cudaError_t create_event_ipc(bool disable_timing,
                             cudaEvent_t* out_local_evt,
                             EventHandle* out_handle) {
  if (!out_local_evt || !out_handle) return cudaErrorInvalidValue;
  unsigned int flags = cudaEventInterprocess;
  if (disable_timing) flags |= cudaEventDisableTiming;

  cudaEvent_t evt;
  auto st = cudaEventCreateWithFlags(&evt, flags);
  if (st != cudaSuccess) return st;

  cudaIpcEventHandle_t eh{};
  st = cudaIpcGetEventHandle(&eh, evt);
  if (st != cudaSuccess) {
    cudaEventDestroy(evt);
    return st;
  }

  *out_local_evt = evt;
  out_handle->evt_handle = eh;
  out_handle->abi_version = 1;
  return cudaSuccess;
}

cudaError_t open_event_ipc(const EventHandle& h, cudaEvent_t* out_evt) {
  if (!out_evt) return cudaErrorInvalidValue;
  cudaEvent_t evt;
  auto st = cudaIpcOpenEventHandle(&evt, h.evt_handle);
  if (st != cudaSuccess) return st;
  *out_evt = evt;
  return cudaSuccess;
}

cudaError_t record_event(cudaEvent_t evt, cudaStream_t stream) {
  // stream==0 则记录在默认流
  return cudaEventRecord(evt, stream);
}

cudaError_t stream_wait_event(cudaStream_t stream, cudaEvent_t evt) {
  return cudaStreamWaitEvent(stream, evt, 0);
}

static inline bool range_src_ok(const RegionInfo& src, uint64_t off, uint64_t n) {
  if (off > src.bytes) return false;
  if (n > src.bytes) return false;
  if (off + n > src.bytes) return false;
  return true;
}

cudaError_t read_async(const RegionInfo& src, void* dst_device,
                       uint64_t bytes, uint64_t src_off,
                       cudaStream_t stream) {
  if (!src.remote_ptr || !dst_device || bytes == 0) return cudaErrorInvalidValue;
  if (!range_src_ok(src, src_off, bytes)) return cudaErrorInvalidValue;
  return cudaMemcpyAsync(dst_device,
                         static_cast<const char*>(src.remote_ptr) + src_off,
                         bytes, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t readv_async(const RegionInfo& src, const ReadDesc* descs, int n) {
  if (!src.remote_ptr || !descs || n <= 0) return cudaErrorInvalidValue;
  for (int i = 0; i < n; ++i) {
    auto st = read_async(src, descs[i].dst, descs[i].bytes, descs[i].src_off, descs[i].stream);
    if (st != cudaSuccess) return st;
  }
  return cudaSuccess;
}

} // namespace wasipc
