# pytests/bench_async_write.py
import os, time, ctypes, ctypes.util
from dataclasses import dataclass
from multiprocessing import get_context
import torch, numpy as np
import pywasipc

BATCH = 2
H = 8192           # Llama3.1-70B hidden size
DTYPE = torch.float16
NUM_ITERS = 200
WARMUP = 20
SLEEP_US = 100     # “MLP 计算”100us

def us_sleep_on_stream(us: int):
    torch.cuda._sleep(int(us * 1_000))  # 近似 us 级别

@dataclass
class IpcPeer:
    inbox_tensor: torch.Tensor
    inbox_region: dict
    inbox_evt_cap: int
    inbox_evt_handle: bytes
    peer_inbox_region: dict
    peer_inbox_evt_cap: int

def setup_rank(local_dev: int, remote_dev: int, peer_conn, rank: int):
    torch.cuda.set_device(local_dev)
    pywasipc.enable_peer_access(local_dev, remote_dev)
    pywasipc.enable_peer_access(remote_dev, local_dev)

    inbox = torch.empty((BATCH, H), dtype=DTYPE, device=f"cuda:{local_dev}")
    handle = pywasipc.register_region(inbox.data_ptr(), inbox.numel() * inbox.element_size())
    cap, evt_handle = pywasipc.create_event_ipc(disable_timing=True)

    my_meta = {
        "dev": local_dev,
        "region_handle": handle["handle"],
        "bytes": handle["bytes"],
        "owner": handle["owner"],
        "evt_handle": evt_handle,
    }

    # 有序握手：rank0 先 send 后 recv；rank1 先 recv 后 send
    if rank == 0:
        peer_conn.send(my_meta)
        meta = peer_conn.recv()
    else:
        meta = peer_conn.recv()
        peer_conn.send(my_meta)

    peer_region = pywasipc.open_region({
        "handle": meta["region_handle"],
        "bytes":  meta["bytes"],
        "owner":  meta["owner"],
    })
    peer_evt = pywasipc.open_event_ipc(meta["evt_handle"])

    return IpcPeer(
        inbox_tensor=inbox,
        inbox_region=handle,
        inbox_evt_cap=cap,
        inbox_evt_handle=evt_handle,
        peer_inbox_region=peer_region,
        peer_inbox_evt_cap=peer_evt,
    )

def run_rank(rank: int, local_dev: int, remote_dev: int, peer_conn, result_queue):
    torch.cuda.set_device(local_dev)
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    peer = setup_rank(local_dev, remote_dev, peer_conn, rank)
    x_out = torch.arange(BATCH * H, device=f"cuda:{local_dev}", dtype=DTYPE).reshape(BATCH, H)

    wait_us, iter_us = [], []
    for it in range(NUM_ITERS):
        i_am_owner = (it % 2 == rank)
        t0 = time.perf_counter()

        if i_am_owner:
            # 等到对端写入可见
            t_wait0 = time.perf_counter()
            pywasipc.stream_wait_event(stream_ptr, peer.inbox_evt_cap)
            stream.synchronize()
            t_wait1 = time.perf_counter()
            wait_us.append((t_wait1 - t_wait0) * 1e6)

            # 本地“MLP”+对端“MLP”
            us_sleep_on_stream(SLEEP_US)
            us_sleep_on_stream(SLEEP_US)

            # 把结果写回并通知对端
            pywasipc.write_async_tensor(peer.peer_inbox_region,
                                        x_out, x_out.numel() * x_out.element_size(), 0, stream_ptr)
            pywasipc.record_event(peer.peer_inbox_evt_cap, stream_ptr)
            stream.synchronize()
        else:
            # 非 owner：先写给对端并通知
            pywasipc.write_async_tensor(peer.peer_inbox_region,
                                        x_out, x_out.numel() * x_out.element_size(), 0, stream_ptr)
            pywasipc.record_event(peer.peer_inbox_evt_cap, stream_ptr)
            stream.synchronize()

            # 再等 owner 回写可见
            pywasipc.stream_wait_event(stream_ptr, peer.inbox_evt_cap)
            stream.synchronize()

        t1 = time.perf_counter()
        iter_us.append((t1 - t0) * 1e6)

    result_queue.put({
        "wait_us": wait_us[WARMUP:],  # 去掉预热
        "iter_us": iter_us[WARMUP:],
        "rank": rank,
    })

def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Need >=2 CUDA GPUs")
        return

    ctx = get_context("spawn")
    # 一条“点对点”Pipe，直接把两个子进程连起来（不是连父进程）
    peer_end0, peer_end1 = ctx.Pipe(duplex=True)
    # 父进程收结果用的队列
    q0, q1 = ctx.Queue(), ctx.Queue()

    p0 = ctx.Process(target=run_rank, args=(0, 0, 1, peer_end0, q0))
    p1 = ctx.Process(target=run_rank, args=(1, 1, 0, peer_end1, q1))
    p0.start(); p1.start()

    s0 = q0.get(); s1 = q1.get()
    p0.join(); p1.join()

    w0 = np.array(s0["wait_us"], dtype=np.float64)
    w1 = np.array(s1["wait_us"], dtype=np.float64)
    i0 = np.array(s0["iter_us"], dtype=np.float64)
    i1 = np.array(s1["iter_us"], dtype=np.float64)

    def stat(x): 
        return dict(avg=float(x.mean()), p50=float(np.percentile(x,50)),
                    p95=float(np.percentile(x,95)), p99=float(np.percentile(x,99)))

    print("== Owner端 stream_wait_event 等待（异步写可见）用时 [微秒] ==")
    if len(w0): print("Rank0 owner:", stat(w0))
    if len(w1): print("Rank1 owner:", stat(w1))

    print("== 每轮总耗时 [微秒]（含 2×100us 计算 + 写/等） ==")
    print("Rank0:", stat(i0))
    print("Rank1:", stat(i1))

if __name__ == "__main__":
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    main()
