#include "wasipc/wasipc.h"
#include <cuda.h>
#include <unistd.h>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>

namespace wasipc {

static std::mutex g_mu;
static std::unordered_set<void*> g_exported; // 仅用于简单生命周期自检

// ---- 事件元信息（仅库内使用）----
struct EventMeta {
  void*        ctrl_dev_ptr = nullptr; // 设备端控制块（u32 epoch）
  int          owner_device  = -1;     // A侧设置
  int          opened_device = -1;     // B侧设置
  int          ctrl_dev      = -1;     // <--- 控制块实际所在GPU（根据 ctrl_dev_ptr 解析）
  cudaStream_t ctrl_stream   = nullptr; // <--- 在 ctrl_dev 上创建的内部流（non-blocking）
  uint32_t     host_epoch    = 0;      // 调用方 record 时自增（或改成 device-side atomic）
  uint32_t     last_seen_ep  = 0;      // wait 端本地计数：下一轮要等到的 epoch
  bool         is_owner      = false;
};


// 以 cudaEvent_t 为键保存本进程内可见的事件元信息
static std::unordered_map<cudaEvent_t, EventMeta> g_evt_meta;

// ---- 工具函数 ----
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

// ---- 事件 IPC ----
cudaError_t create_event_ipc(bool disable_timing,
                             cudaEvent_t* out_local_evt,
                             EventHandle* out_handle) {
  if (!out_local_evt || !out_handle) return cudaErrorInvalidValue;
  unsigned int flags = cudaEventInterprocess;
  if (disable_timing) flags |= cudaEventDisableTiming;

  int dev = -1;
  cudaGetDevice(&dev);

  // 1) 创建事件
  cudaEvent_t evt;
  auto st = cudaEventCreateWithFlags(&evt, flags);
  if (st != cudaSuccess) return st;

  // 2) 创建控制块（u32 epoch = 0）
  void* ctrl_dev_ptr = nullptr;
  st = cudaMalloc(&ctrl_dev_ptr, sizeof(uint32_t));
  if (st != cudaSuccess) {
    cudaEventDestroy(evt);
    return st;
  }
  st = cudaMemset(ctrl_dev_ptr, 0, sizeof(uint32_t));
  if (st != cudaSuccess) {
    cudaFree(ctrl_dev_ptr);
    cudaEventDestroy(evt);
    return st;
  }

  // 3) 导出句柄
  cudaIpcEventHandle_t eh{};
  st = cudaIpcGetEventHandle(&eh, evt);
  if (st != cudaSuccess) {
    cudaFree(ctrl_dev_ptr);
    cudaEventDestroy(evt);
    return st;
  }

  cudaIpcMemHandle_t mh{};
  st = cudaIpcGetMemHandle(&mh, ctrl_dev_ptr);
  if (st != cudaSuccess) {
    cudaFree(ctrl_dev_ptr);
    cudaEventDestroy(evt);
    return st;
  }

  // 4) 输出句柄（ABI=2）
  out_handle->evt_handle  = eh;
  out_handle->ctrl_mem    = mh;
  out_handle->abi_version = 2;

  *out_local_evt = evt;

  // 5) 记录 A 侧元信息
  {
    std::lock_guard<std::mutex> lk(g_mu);
    EventMeta meta;
    meta.ctrl_dev_ptr = ctrl_dev_ptr;
    meta.owner_device = dev;
    meta.opened_device= dev;
    meta.ctrl_dev     = dev;  // 控制块就分配在本GPU
    int prev = -1; cudaGetDevice(&prev);
    if (prev != meta.ctrl_dev) cudaSetDevice(meta.ctrl_dev);
    cudaStreamCreateWithFlags(&meta.ctrl_stream, cudaStreamNonBlocking);
    if (prev != meta.ctrl_dev) cudaSetDevice(prev);
    meta.host_epoch   = 0;
    meta.last_seen_ep = 0;
    meta.is_owner     = true;
    g_evt_meta[evt]   = meta;
}


  return cudaSuccess;
}

// 事件打开：B侧
// - 打开事件句柄与控制块（使用 cudaIpcMemLazyEnablePeerAccess）
// - 记录本地元信息（is_owner=false）
cudaError_t open_event_ipc(const EventHandle& h, cudaEvent_t* out_evt) {
  if (!out_evt) return cudaErrorInvalidValue;
  if (h.abi_version != 2) return cudaErrorInvalidValue; // 本版本仅支持 v2

  cudaEvent_t evt;
  auto st = cudaIpcOpenEventHandle(&evt, h.evt_handle);
  if (st != cudaSuccess) return st;

  int cur = -1; cudaGetDevice(&cur);

  void* ctrl_dev_ptr = nullptr;
  st = cudaIpcOpenMemHandle(&ctrl_dev_ptr, h.ctrl_mem, cudaIpcMemLazyEnablePeerAccess);
  if (st != cudaSuccess) {
    // 回滚事件对象
    cudaEventDestroy(evt);
    return st;
  }

  *out_evt = evt;

  cudaPointerAttributes attr{};
  cudaPointerGetAttributes(&attr, ctrl_dev_ptr);
  int ctrl_dev = -1;
  #if CUDART_VERSION >= 11000
    if (attr.type == cudaMemoryTypeDevice) ctrl_dev = attr.device;
  #else
    if (attr.memoryType == cudaMemoryTypeDevice) ctrl_dev = attr.device;
  #endif

  *out_evt = evt;

  {
    std::lock_guard<std::mutex> lk(g_mu);
    EventMeta meta;
    meta.ctrl_dev_ptr = ctrl_dev_ptr;
    meta.opened_device= cur;
    meta.ctrl_dev     = ctrl_dev;  // 可能是 0（owner 的 GPU）
    int prev = -1; cudaGetDevice(&prev);
    if (meta.ctrl_dev >= 0 && prev != meta.ctrl_dev) cudaSetDevice(meta.ctrl_dev);
    cudaStreamCreateWithFlags(&meta.ctrl_stream, cudaStreamNonBlocking);
    if (meta.ctrl_dev >= 0 && prev != meta.ctrl_dev) cudaSetDevice(prev);
    meta.host_epoch   = 0;
    meta.last_seen_ep = 0;
    meta.is_owner     = false;
    g_evt_meta[evt]   = meta;
  }


  return cudaSuccess;
}

cudaError_t record_event(cudaEvent_t evt, cudaStream_t stream) {
  // 1) 记录“本轮”事件（调用方的流：通常在写数据之后）
  cudaError_t st = cudaEventRecord(evt, stream);
  if (st != cudaSuccess) return st;

  // 2) 取元信息并自增 epoch（谁 record 谁负责自增）
  void*     ctrl_dev_ptr = nullptr;
  int       ctrl_dev     = -1;
  cudaStream_t ctrl_stream = nullptr;
  uint32_t  v = 0;

  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_evt_meta.find(evt);
    if (it == g_evt_meta.end()) return cudaErrorInvalidResourceHandle;
    if (!it->second.ctrl_dev_ptr || !it->second.ctrl_stream || it->second.ctrl_dev < 0) {
      // 缺控制块/内部流：退化为仅 record（不建议，但保持健壮）
      return cudaSuccess;
    }
    it->second.host_epoch += 1;
    v            = it->second.host_epoch;
    ctrl_dev_ptr = it->second.ctrl_dev_ptr;
    ctrl_dev     = it->second.ctrl_dev;
    ctrl_stream  = it->second.ctrl_stream;
  }

  // 3) 在“控制块所在 GPU”的内部流上：先等本轮事件，再写 epoch（保证顺序：写数据→record→(wait)→写epoch）
  int prev = -1; cudaGetDevice(&prev);
  if (prev != ctrl_dev) cudaSetDevice(ctrl_dev);

  st = cudaStreamWaitEvent(ctrl_stream, evt, 0);   // 跨设备等待“本轮”record
  if (st == cudaSuccess) {
    st = cudaMemcpyAsync(ctrl_dev_ptr, &v, sizeof(uint32_t),
                         cudaMemcpyHostToDevice, ctrl_stream);
  }

  if (prev != ctrl_dev) cudaSetDevice(prev);
  return st;
}

cudaError_t stream_wait_event(cudaStream_t stream, cudaEvent_t evt) {
  void*     ctrl_dev_ptr = nullptr;
  int       ctrl_dev     = -1;
  uint32_t  expected     = 0;

  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_evt_meta.find(evt);
    if (it == g_evt_meta.end()) return cudaErrorInvalidResourceHandle;
    it->second.last_seen_ep += 1;
    expected     = it->second.last_seen_ep;
    ctrl_dev_ptr = it->second.ctrl_dev_ptr;
    ctrl_dev     = it->second.ctrl_dev;
  }

  if (ctrl_dev_ptr) {
    // 在控制块所在 GPU 上轮询直到 >= expected
    int prev = -1; cudaGetDevice(&prev);
    if (ctrl_dev >= 0 && prev != ctrl_dev) cudaSetDevice(ctrl_dev);

    uint32_t cur = 0;
    constexpr int kSpinItersBusy = 200;
    constexpr int kSleepUs = 20;
    int spins = 0;
    cudaError_t st = cudaSuccess;

    while (true) {
      st = cudaMemcpy(&cur, ctrl_dev_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess) break;
      if (cur >= expected) break;
      if (spins < kSpinItersBusy) ++spins;
      else {
        #if __cplusplus >= 201103L
          std::this_thread::sleep_for(std::chrono::microseconds(kSleepUs));
        #else
          usleep(kSleepUs);
        #endif
      }
    }

    if (ctrl_dev >= 0 && prev != ctrl_dev) cudaSetDevice(prev);
    if (st != cudaSuccess) return st;
  }

  // 在调用方给定的 stream 上，追加一次 GPU 级等待（把后续访问与“本轮 record”关联起来）
  return cudaStreamWaitEvent(stream, evt, 0);
}

} // namespace wasipc
