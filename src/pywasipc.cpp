// SPDX-License-Identifier: Apache-2.0
#include "wasipc/wasipc.h"

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using wasipc::RegionHandle;
using wasipc::RegionInfo;

static inline cudaStream_t as_stream(uint64_t s) {
  return reinterpret_cast<cudaStream_t>(s);
}

static inline void throw_if(cudaError_t st) {
  if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
}

static inline void* obj_data_ptr(py::handle obj) {
  // Duck-typing 调用 obj.data_ptr() -> int/long
  if (!py::hasattr(obj, "data_ptr")) {
    throw std::runtime_error("object has no .data_ptr(); pass a PyTorch CUDA tensor or raw addr");
  }
  py::object addr_obj = obj.attr("data_ptr")();
  uint64_t addr = addr_obj.cast<uint64_t>();
  return reinterpret_cast<void*>(addr);
}

PYBIND11_MODULE(pywasipc, m) {
  m.doc() = "CUDA IPC helper (no libtorch headers)";

  // ---- 基础能力 ----
  m.def("can_access_peer", &wasipc::can_access_peer,
        py::arg("req_dev"), py::arg("owner_dev"));
  m.def("enable_peer_access",
        [](int req_dev, int owner_dev) {
          auto st = wasipc::enable_peer_access(req_dev, owner_dev);
          if (st != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(st));
          }
        },
        py::arg("req_dev"), py::arg("owner_dev"));

  // RegionInfo 暴露只读字段
  py::class_<RegionInfo>(m, "RegionInfo")
      .def_readonly("remote_ptr", &RegionInfo::remote_ptr)
      .def_readonly("owner_device", &RegionInfo::owner_device)
      .def_readonly("bytes", &RegionInfo::bytes);

  // 注册/注销区域（owner 进程）
  m.def("register_region",
        [](uint64_t device_ptr, uint64_t bytes) {
          RegionHandle h{};
          void* ptr = reinterpret_cast<void*>(device_ptr);
          auto st = wasipc::register_region(ptr, bytes, &h);
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
          py::dict d;
          d["handle"] = py::bytes(reinterpret_cast<const char*>(&h.mem_handle), sizeof(h.mem_handle));
          d["owner"]  = h.owner_device;
          d["bytes"]  = py::int_(h.bytes);
          return d;
        },
        py::arg("device_ptr"), py::arg("bytes"));

  m.def("deregister_region",
        [](uint64_t device_ptr) {
          void* ptr = reinterpret_cast<void*>(device_ptr);
          auto st = wasipc::deregister_region(ptr);
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("device_ptr"));

  // 打开/关闭远端区域（requester 进程）
  m.def("open_region",
        [](py::dict h) {
          if (!h.contains("handle") || !h.contains("bytes") || !h.contains("owner")) {
            throw std::runtime_error("open_region expects dict with keys: handle, bytes, owner");
          }
          RegionHandle rh{};
          std::string blob = py::cast<std::string>(h["handle"]);
          if (blob.size() != sizeof(rh.mem_handle)) {
            throw std::runtime_error("invalid handle blob size");
          }
          memcpy(&rh.mem_handle, blob.data(), sizeof(rh.mem_handle));
          rh.bytes        = py::cast<uint64_t>(h["bytes"]);
          rh.owner_device = py::cast<int>(h["owner"]);
          rh.abi_version  = 1;

          RegionInfo info{};
          auto st = wasipc::open_region(rh, &info, cudaIpcMemLazyEnablePeerAccess);
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
          return info;
        },
        py::arg("handle_dict"));
    
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
        [](RegionInfo& info) {
          auto st = wasipc::close_region(&info);
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("info"));

  // ---- 读接口 ----
  m.def("read_async",
        [](const RegionInfo& src, uint64_t dst_device_ptr, uint64_t bytes,
           uint64_t src_off, uint64_t stream) {
          auto st = wasipc::read_async(src,
                                       reinterpret_cast<void*>(dst_device_ptr),
                                       bytes, src_off, as_stream(stream));
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("src"), py::arg("dst_device_ptr"), py::arg("bytes"),
        py::arg("src_off"), py::arg("stream"));

  m.def("read_async_into_tensor",
        [](const RegionInfo& src, py::object tensor, uint64_t bytes,
           uint64_t src_off, uint64_t stream) {
          void* dst = obj_data_ptr(tensor);
          auto st = wasipc::read_async(src, dst, bytes, src_off, as_stream(stream));
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("src"), py::arg("tensor"), py::arg("bytes"),
        py::arg("src_off"), py::arg("stream"));

  m.def("readv_async",
        [](const RegionInfo& src, std::vector<std::tuple<uint64_t,uint64_t,uint64_t,uint64_t>> descs) {
          std::vector<wasipc::ReadDesc> v;
          v.reserve(descs.size());
          for (auto &t : descs) {
            wasipc::ReadDesc d;
            d.dst     = reinterpret_cast<void*>(std::get<0>(t));
            d.bytes   = std::get<1>(t);
            d.src_off = std::get<2>(t);
            d.stream  = as_stream(std::get<3>(t));
            v.push_back(d);
          }
          auto st = wasipc::readv_async(src, v.data(), static_cast<int>(v.size()));
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("src"), py::arg("descs"));

  // ---- 写接口 ----
  m.def("write_async",
        [](const RegionInfo& dst_remote, uint64_t src_device_ptr, uint64_t bytes,
           uint64_t dst_off, uint64_t stream) {
          if (!dst_remote.remote_ptr) throw std::runtime_error("invalid remote ptr");
          auto st = cudaMemcpyAsync(
              static_cast<char*>(dst_remote.remote_ptr) + dst_off,
              reinterpret_cast<void*>(src_device_ptr),
              bytes, cudaMemcpyDeviceToDevice, as_stream(stream));
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("dst_remote"), py::arg("src_device_ptr"), py::arg("bytes"),
        py::arg("dst_off"), py::arg("stream"));

  m.def("write_async_tensor",
        [](const RegionInfo& dst_remote, py::object tensor, uint64_t bytes,
           uint64_t dst_off, uint64_t stream) {
          if (!dst_remote.remote_ptr) throw std::runtime_error("invalid remote ptr");
          void* src = obj_data_ptr(tensor);
          auto st = cudaMemcpyAsync(
              static_cast<char*>(dst_remote.remote_ptr) + dst_off,
              src, bytes, cudaMemcpyDeviceToDevice, as_stream(stream));
          if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        },
        py::arg("dst_remote"), py::arg("tensor"), py::arg("bytes"),
        py::arg("dst_off"), py::arg("stream"));

  m.def("writev_async_tensor",
        [](const RegionInfo& dst_remote,
           std::vector<std::tuple<py::object,uint64_t,uint64_t>> descs,
           uint64_t stream) {
          if (!dst_remote.remote_ptr) throw std::runtime_error("invalid remote ptr");
          for (auto &t : descs) {
            void* src = obj_data_ptr(std::get<0>(t));
            uint64_t bytes = std::get<1>(t);
            uint64_t off   = std::get<2>(t);
            auto st = cudaMemcpyAsync(
                static_cast<char*>(dst_remote.remote_ptr) + off,
                src, bytes, cudaMemcpyDeviceToDevice, as_stream(stream));
            if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
          }
        },
        py::arg("dst_remote"), py::arg("descs"), py::arg("stream"));

  // ---- 事件 IPC ----
  struct EventCap { cudaEvent_t evt{nullptr}; };
  py::class_<EventCap>(m, "EventCap");

  // 使用 wasipc 的 create_event_ipc：返回 (cap, handle_bytes)
  m.def("create_event_ipc",
        [](bool disable_timing) {
          cudaEvent_t local_evt{};
          wasipc::EventHandle h{};              // 注意：这是 wasipc.h 里定义的 struct
          throw_if(wasipc::create_event_ipc(disable_timing, &local_evt, &h));
          EventCap cap; cap.evt = local_evt;
          // 将 EventHandle 整体序列化为 bytes，跨进程传递
          py::bytes blob(reinterpret_cast<const char*>(&h), sizeof(h));
          return py::make_tuple(std::move(cap), std::move(blob));
        },
        py::arg("disable_timing") = true);

  m.def("open_event_ipc",
        [](py::bytes handle_blob) {
          std::string blob = handle_blob;
          if (blob.size() != sizeof(wasipc::EventHandle))
            throw std::runtime_error("invalid wasipc::EventHandle bytes size");
          wasipc::EventHandle h{};
          memcpy(&h, blob.data(), sizeof(h));
          cudaEvent_t e{};
          throw_if(wasipc::open_event_ipc(h, &e));   // 打开事件 + 内部控制块
          EventCap cap; cap.evt = e;
          return cap;
        },
        py::arg("handle"));

  m.def("record_event",
        [](const EventCap& cap, uint64_t stream) {
          // 走 wasipc 版本：会在内部按顺序更新 epoch，防止电平粘连
          throw_if(wasipc::record_event(cap.evt, as_stream(stream)));
        },
        py::arg("cap"), py::arg("stream"));

  m.def("stream_wait_event",
        [](uint64_t stream, const EventCap& cap) {
          // 走 wasipc 版本：边沿语义等待“下一轮”，不会被旧完成态满足
          throw_if(wasipc::stream_wait_event(as_stream(stream), cap.evt));
        },
        py::arg("stream"), py::arg("cap"));

}
