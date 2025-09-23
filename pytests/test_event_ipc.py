# pytests/test_event_ipc.py
import ctypes, ctypes.util
import torch
import pywasipc
import pytest
from multiprocessing import get_context

def _owner_proc(pipe):
    torch.cuda.set_device(0)
    B, H = 8, 16
    dtype = torch.float16
    inbox = torch.full((B, H), -1.0, dtype=dtype, device="cuda:0")

    # export CUDA IPC mem handle（owner -> peer）
    libcudart = ctypes.CDLL(ctypes.util.find_library("cudart"))
    HANDLE_SZ = 64
    hbuf = (ctypes.c_char * HANDLE_SZ)()
    rc = libcudart.cudaIpcGetMemHandle(hbuf, ctypes.c_void_p(inbox.data_ptr()))
    assert rc == 0, "cudaIpcGetMemHandle failed"
    region_handle = bytes(hbuf)

    # 事件：owner 创建，peer 打开并记录
    cap, evt_handle = pywasipc.create_event_ipc(disable_timing=True)

    pipe.send({
        "region_handle": region_handle,
        "bytes": int(inbox.element_size() * inbox.numel()),
        "owner": 0,
        "shape": (B, H),
        "dtype": str(dtype),
        "evt_handle": evt_handle,
    })

    stream_ptr = torch.cuda.current_stream().cuda_stream
    pywasipc.stream_wait_event(stream_ptr, cap)  # 等对端 record_event
    torch.cuda.current_stream().synchronize()

    expect = torch.arange(B * H, dtype=dtype, device="cuda:0").reshape(B, H)
    assert torch.allclose(inbox, expect), "owner inbox mismatch after peer write"
    pipe.send("OK")

def _peer_proc(pipe):
    torch.cuda.set_device(1)
    meta = pipe.recv()
    B, H = meta["shape"]
    dtype = torch.float16

    region = pywasipc.open_region({
        "handle": meta["region_handle"],
        "bytes": meta["bytes"],
        "owner": meta["owner"],
    })
    evt_peer = pywasipc.open_event_ipc(meta["evt_handle"])

    x = torch.arange(B * H, dtype=dtype, device="cuda:1").reshape(B, H)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    pywasipc.write_async_tensor(region, x, x.numel() * x.element_size(), 0, stream_ptr)
    pywasipc.record_event(evt_peer, stream_ptr)
    torch.cuda.current_stream().synchronize()

    assert pipe.recv() == "OK"

def _has_two_gpus():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2

@pytest.mark.skipif(not _has_two_gpus(), reason="need >=2 CUDA GPUs")
def test_event_ipc_notify_and_remote_write_roundtrip():
    """
    双进程严格测试事件 IPC 顺序：
      Owner(0): 分配 inbox，export IPC mem handle；create_event_ipc；把句柄发给 Peer；
                在自己的流上 stream_wait_event(evt)；随后校验 inbox 内容
      Peer(1):  open_region + open_event_ipc；write_async_tensor；record_event(evt)
    """
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    p0 = ctx.Process(target=_owner_proc, args=(parent_conn,))
    p1 = ctx.Process(target=_peer_proc,  args=(child_conn,))
    p0.start(); p1.start()
    p0.join(10.0); p1.join(10.0)
    assert p0.exitcode == 0 and p1.exitcode == 0
