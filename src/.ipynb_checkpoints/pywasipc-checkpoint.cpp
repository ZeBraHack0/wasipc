#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include "wasipc/wasipc.h"

namespace py = pybind11;
using namespace wasipc;

static inline cudaStream_t as_stream(std::uintptr_t p) {
  return reinterpret_cast<cudaStream_t>(p);
}
static inline void throw_if(cudaError_t st) {
  if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
}

PYBIND11_MODULE(pywasipc, m) {
  m.doc() = "CUDA IPC + P2P one-sided GPU memory read (WAS IPC)";

  // ----- POD 封装（轻量，只读属性） -----
  py::class_<RegionInfo>(m, "RegionInfo")
      .def_property_readonly("remote_ptr",
        [](const RegionInfo& r){ return reinterpret_cast<std::uintptr_t>(r.remote_ptr); })
      .def_readonly("owner_device", &RegionInfo::owner_device)
      .def_readonly("bytes", &RegionInfo::bytes);

  // ----- owner 侧：注册/注销 -----
  m.def("register_region",
        [](std::uintptr_t device_ptr, std::uint64_t bytes)->py::bytes {
          RegionHandle h{};
          {
            // 极短操作，不释放 GIL；
            void* p = reinterpret_cast<void*>(device_ptr);
            throw_if(register_region(p, bytes, &h));
          }
          auto v = serialize(h);
          return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        },
        py::arg("device_ptr"), py::arg("bytes"),
        R"pbdoc(
        Owner side: export a device buffer (cudaMalloc) as a handle (bytes).
        device_ptr: int(pointer from e.g. torch_tensor.data_ptr()).
        )pbdoc");

  m.def("deregister_region",
        [](std::uintptr_t device_ptr){
          void* p = reinterpret_cast<void*>(device_ptr);
          throw_if(deregister_region(p));
        },
        py::arg("device_ptr"));

  // ----- requester 侧：打开/关闭 -----
  m.def("open_region",
        [](py::bytes handle_bytes)->RegionInfo {
          std::string s = handle_bytes; // copy bytes -> std::string
          RegionHandle h{};
          if (!deserialize(s.data(), s.size(), &h))
            throw std::runtime_error("invalid handle bytes");
          RegionInfo info{};
          throw_if(open_region(h, &info, cudaIpcMemLazyEnablePeerAccess));
          return info;
        },
        py::arg("handle_bytes"),
        R"pbdoc(
        Requester side: open handle bytes to get a RegionInfo (remote_ptr, bytes, owner_device).
        )pbdoc");

  m.def("close_region",
        [](RegionInfo& info){ throw_if(close_region(&info)); },
        py::arg("info"));

  // ----- 读取：单次/批量（异步） -----
  m.def("read_async",
        [](const RegionInfo& info, std::uintptr_t dst_device,
           std::uint64_t nbytes, std::uint64_t src_off, std::uintptr_t stream_ptr){
          void* dst = reinterpret_cast<void*>(dst_device);
          cudaStream_t s = as_stream(stream_ptr);
          // 释放 GIL，避免小量同步开销阻塞 Python
          py::gil_scoped_release rel;
          throw_if(read_async(info, dst, nbytes, src_off, s));
        },
        py::arg("info"), py::arg("dst_device"), py::arg("bytes"),
        py::arg("src_off")=0, py::arg("stream_ptr")=0);

  m.def("readv_async",
        [](const RegionInfo& info,
           const std::vector<std::tuple<std::uintptr_t, std::uint64_t, std::uint64_t, std::uintptr_t>>& descs){
          std::vector<ReadDesc> v; v.reserve(descs.size());
          for (auto& t : descs) {
            ReadDesc d{};
            d.dst = reinterpret_cast<void*>(std::get<0>(t));
            d.bytes   = std::get<1>(t);
            d.src_off = std::get<2>(t);
            d.stream  = as_stream(std::get<3>(t));
            v.push_back(d);
          }
          py::gil_scoped_release rel;
          throw_if(readv_async(info, v.data(), static_cast<int>(v.size())));
        },
        py::arg("info"), py::arg("descs"),
        R"pbdoc(
        descs: list of tuples (dst_ptr:int, bytes:int, src_off:int, stream_ptr:int)
        )pbdoc");

  // ----- 辅助：peer 能力/启用 -----
  m.def("can_access_peer", &can_access_peer, py::arg("requester_dev"), py::arg("owner_dev"));
  m.def("enable_peer_access",
        [](int requester_dev, int owner_dev){
          throw_if(enable_peer_access(requester_dev, owner_dev));
        },
        py::arg("requester_dev"), py::arg("owner_dev"));
}
