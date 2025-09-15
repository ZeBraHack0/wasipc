import os
import time
import pytest
import torch
import pywasipc
import multiprocessing as mp

# 避免 fork + CUDA 带来的隐患
mp.set_start_method("spawn", force=True)

OWNER_ENV = "WASIPC_OWNER_DEV"
REQ_ENV   = "WASIPC_REQ_DEV"

def _pick_devices():
    owner = int(os.getenv(OWNER_ENV, "0"))
    req   = int(os.getenv(REQ_ENV,   "1"))
    return owner, req

def _have_two_gpus():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2

@pytest.fixture(scope="session")
def devices():
    if not _have_two_gpus():
        pytest.skip("CUDA or >=2 GPUs not available")
    owner, req = _pick_devices()
    if owner == req:
        pytest.skip("owner and requester must be different GPUs")
    if not pywasipc.can_access_peer(req, owner):
        pytest.skip(f"P2P not available between dev{req} and dev{owner}")
    pywasipc.enable_peer_access(req, owner)
    return owner, req

@pytest.fixture(scope="function")
def sizes():
    # 区域/页尺寸（跑得快）
    REGION_BYTES = 32 * (1 << 20)  # 32 MiB
    PAGE_BYTES   = 8  * (1 << 20)  # 8  MiB
    return REGION_BYTES, PAGE_BYTES

# 生成同样的张量；owner/requester 两侧都能独立重建用于校验
def make_pattern(count_bytes: int, offset: int = 0, device: str = "cpu"):
    # 简单 uint8 张量：(i+offset)*7 mod 256
    i0 = offset
    t = torch.arange(i0, i0 + count_bytes, dtype=torch.int64, device=device)
    return (t.to(torch.uint8) * 7).to(torch.uint8)

def _owner_proc(dev: int, region_bytes: int, q_handle: mp.Queue, ev_stop: mp.Event):
    import torch, pywasipc
    torch.cuda.set_device(dev)
    # 分配并写入张量
    buf = torch.empty(region_bytes, dtype=torch.uint8, device=f"cuda:{dev}")
    pat = make_pattern(region_bytes, 0, device=f"cuda:{dev}")
    buf.copy_(pat)
    # 注册并把句柄发回
    handle = pywasipc.register_region(buf.data_ptr(), region_bytes)
    q_handle.put(handle)
    # 阻塞等待测试结束
    ev_stop.wait()
    # 清理
    pywasipc.deregister_region(buf.data_ptr())
    del buf
    torch.cuda.synchronize(dev)

@pytest.fixture(scope="function")
def owner_process(devices, sizes):
    owner, _ = devices
    REGION_BYTES, _ = sizes
    q_handle = mp.Queue(maxsize=1)
    ev_stop  = mp.Event()
    proc = mp.Process(target=_owner_proc, args=(owner, REGION_BYTES, q_handle, ev_stop), daemon=True)
    proc.start()
    try:
        handle = q_handle.get(timeout=10.0)
    except Exception:
        proc.terminate(); proc.join(timeout=2.0)
        raise RuntimeError("owner process did not produce a handle in time")
    yield {"handle": handle, "stop": ev_stop, "proc": proc}
    # 通知 owner 退出并收尸
    ev_stop.set()
    proc.join(timeout=5.0)

@pytest.fixture(scope="function")
def requester_info(devices, owner_process):
    _, req = devices
    torch.cuda.set_device(req)
    info = pywasipc.open_region(owner_process["handle"])
    yield info
    pywasipc.close_region(info)
    torch.cuda.synchronize(req)

# 让测试可以重建期望片段
@pytest.fixture(scope="function")
def expect_slice():
    def _f(n_bytes: int, src_off: int, device: str):
        return make_pattern(n_bytes, offset=src_off, device=device)
    return _f
