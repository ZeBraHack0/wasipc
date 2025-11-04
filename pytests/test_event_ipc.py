# pytests/test_event_ipc.py
import ctypes, ctypes.util
import torch
import pywasipc
import pytest
from multiprocessing import get_context
import time

ROUNDS_WAIT_FIRST = 5   # 场景A：Owner先wait，Peer后record
ROUNDS_RECORD_FIRST = 5 # 场景B：Peer先record，Owner后wait

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

    # ========= 场景A：Owner先wait，Peer后record（验证“等在前，不会粘连”） =========
    for r in range(ROUNDS_WAIT_FIRST):
        # 让Peer做好准备，但不进行record，确保Owner这边先进入等待
        pipe.send(("ARM", r))
        assert pipe.recv() == ("ACK_ARM", r)

        # Owner先等这一轮（如果有粘连，这里会被上一轮满足而过早返回）
        pywasipc.stream_wait_event(stream_ptr, cap)
        torch.cuda.current_stream().synchronize()

        expect = (torch.arange(B * H, dtype=dtype, device="cuda:0")
                    .reshape(B, H) + r)
        assert torch.allclose(inbox, expect), f"[A] round {r} mismatch"
        pipe.send(("OK", r))

    # ========= 场景B：Peer先record，Owner后wait（验证“记在前，也严格受控”） =========
    for r in range(ROUNDS_WAIT_FIRST, ROUNDS_WAIT_FIRST + ROUNDS_RECORD_FIRST):
        # Peer先完成write+record后再告诉Owner
        tag, ridx = pipe.recv()
        assert tag == "FIRE" and ridx == r


        # Owner随后再等
        pywasipc.stream_wait_event(stream_ptr, cap)
        torch.cuda.current_stream().synchronize()

        expect = (torch.arange(B * H, dtype=dtype, device="cuda:0")
                    .reshape(B, H) + r)
        assert torch.allclose(inbox, expect), f"[B] round {r} mismatch"
        pipe.send(("OK", r))

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

    stream_ptr = torch.cuda.current_stream().cuda_stream

    # ========= 场景A：Owner先wait，Peer后record =========
    for r in range(ROUNDS_WAIT_FIRST):
        msg = pipe.recv()
        assert msg == ("ARM", r)
        pipe.send(("ACK_ARM", r))
        time.sleep(1)

        # 生成本轮数据并写入，然后record
        x = (torch.arange(B * H, dtype=dtype, device="cuda:1")
                .reshape(B, H) + r)
        pywasipc.write_async_tensor(
            region, x, x.numel() * x.element_size(), 0, stream_ptr
        )
        pywasipc.record_event(evt_peer, stream_ptr)
        torch.cuda.current_stream().synchronize()

        # 等Owner确认收到并校验通过
        assert pipe.recv() == ("OK", r)

    # ========= 场景B：Peer先record，Owner后wait =========
    for r in range(ROUNDS_WAIT_FIRST, ROUNDS_WAIT_FIRST + ROUNDS_RECORD_FIRST):
        x = (torch.arange(B * H, dtype=dtype, device="cuda:1")
                .reshape(B, H) + r)
        pywasipc.write_async_tensor(
            region, x, x.numel() * x.element_size(), 0, stream_ptr
        )
        pywasipc.record_event(evt_peer, stream_ptr)
        torch.cuda.current_stream().synchronize()

        # 告诉Owner这一轮已经record完成，让Owner再去wait
        pipe.send(("FIRE", r))
        assert pipe.recv() == ("OK", r)

def _has_two_gpus():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2

@pytest.mark.skipif(not _has_two_gpus(), reason="need >=2 CUDA GPUs")
def test_event_ipc_notify_and_remote_write_multiround():
    """
    多轮双进程测试同一个 IPC 事件的重复使用，覆盖两种时序：
      场景A：Owner先wait，Peer后record（验证不粘连）
      场景B：Peer先record，Owner后wait（验证严格顺序）
    """
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    p0 = ctx.Process(target=_owner_proc, args=(parent_conn,))
    p1 = ctx.Process(target=_peer_proc,  args=(child_conn,))
    p0.start(); p1.start()
    p0.join(30.0); p1.join(30.0)
    assert p0.exitcode == 0 and p1.exitcode == 0
