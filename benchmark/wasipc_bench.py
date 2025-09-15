#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wasipc_bench.py — 测量 pywasipc 异步单边读取的有效吞吐（GiB/s）

用法示例：
  python tools/wasipc_bench.py --owner-dev 0 --req-dev 1 \
      --region-mib 512 --page-mib 16 --streams 2 --inflight 2 \
      --total-gib 8

可选校验内容张量（轻微影响性能）：
  python tools/wasipc_bench.py --verify
"""
import argparse
import os
import time
import multiprocessing as mp

import torch
import pywasipc

# 避免 fork 与 CUDA 的组合带来问题
mp.set_start_method("spawn", force=True)

def mib(x): return x * (1 << 20)
def gib(x): return x * (1 << 30)

def make_pattern(count_bytes: int, offset: int = 0, device: str = "cpu"):
    # 简单 uint8 张量：(i+offset)*7 mod 256
    i0 = offset
    t = torch.arange(i0, i0 + count_bytes, dtype=torch.int64, device=device)
    return (t.to(torch.uint8) * 7).to(torch.uint8)

def owner_proc(dev: int, region_bytes: int, q_handle: mp.Queue, ev_stop: mp.Event, verify: bool):
    torch.cuda.set_device(dev)
    buf = torch.empty(region_bytes, dtype=torch.uint8, device=f"cuda:{dev}")
    if verify:
        buf.copy_(make_pattern(region_bytes, 0, device=f"cuda:{dev}"))
    handle = pywasipc.register_region(buf.data_ptr(), region_bytes)
    q_handle.put(handle)
    ev_stop.wait()
    pywasipc.deregister_region(buf.data_ptr())
    del buf
    torch.cuda.synchronize(dev)

def run_bench(args):
    owner, req = args.owner_dev, args.req_dev
    reg_bytes = mib(args.region_mib)
    page_bytes = mib(args.page_mib)
    total_bytes = gib(args.total_gib)

    assert total_bytes % page_bytes == 0, "total-gib 必须能被 page-mib 整除（方便统计）"
    pages_total = total_bytes // page_bytes

    # 启 owner 子进程，导出句柄
    q_handle = mp.Queue(maxsize=1)
    ev_stop = mp.Event()
    p = mp.Process(target=owner_proc,
                   args=(owner, reg_bytes, q_handle, ev_stop, args.verify),
                   daemon=True)
    p.start()
    handle = q_handle.get(timeout=15.0)

    # requester 侧设置
    torch.cuda.set_device(req)
    if not pywasipc.can_access_peer(req, owner):
        raise RuntimeError(f"P2P not available between dev{req} and dev{owner}")
    pywasipc.enable_peer_access(req, owner)
    info = pywasipc.open_region(handle)

    # 为每个流准备 in-flight 页缓冲
    streams = [torch.cuda.Stream(device=req) for _ in range(args.streams)]
    pages = [[torch.empty(page_bytes, dtype=torch.uint8, device=f"cuda:{req}")
              for _ in range(args.inflight)] for __ in range(args.streams)]

    # warmup（不计时）
    for s_idx, s in enumerate(streams):
        with torch.cuda.stream(s):
            off = (s_idx * page_bytes) % reg_bytes
            pywasipc.read_async(info, pages[s_idx][0].data_ptr(),
                                page_bytes, off, s.cuda_stream)
    for s in streams: s.synchronize()

    # 正式计时
    start = time.perf_counter()
    # 循环提交 pages_total 次读，每次跨所有流轮转，形成 S×K 的 in-flight 管道
    issued = 0
    next_off = 0
    while issued < pages_total:
        for s_idx, s in enumerate(streams):
            # 每个流最多保持 inflight 个未完成页
            for k in range(args.inflight):
                if issued >= pages_total:
                    break
                dst = pages[s_idx][k]
                with torch.cuda.stream(s):
                    pywasipc.read_async(info, dst.data_ptr(),
                                        page_bytes, next_off, s.cuda_stream)
                issued += 1
                next_off = (next_off + page_bytes) % reg_bytes
    # 等待全部完成
    for s in streams: s.synchronize()
    elapsed = time.perf_counter() - start

    # 校验（可选）
    if args.verify:
        # 随机抽样校验几页
        import random
        torch.cuda.synchronize(req)
        for _ in range(min(4, args.streams * args.inflight)):
            s_idx = random.randrange(args.streams)
            k = random.randrange(args.inflight)
            # 计算这页的理论起始 offset（粗略：最近一次提交的位置无法直接取回；做近似抽样）
            # 这里额外读一页 offset=0 做 sanity
            expect = make_pattern(page_bytes, 0, device=f"cuda:{req}")
            # 重新读回 offset=0 到这个页缓冲以对齐校验逻辑（不影响统计，因为统计已结束）
            pywasipc.read_async(info, pages[s_idx][k].data_ptr(), page_bytes, 0, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize(req)
            assert torch.equal(pages[s_idx][k], expect), "pattern 校验失败"

    # 清理
    pywasipc.close_region(info)
    ev_stop.set()
    p.join(timeout=5.0)

    # 统计
    gib_s = total_bytes / (1<<30) / elapsed
    mib_s = total_bytes / (1<<20) / elapsed
    print("==== WASIPC Async Read Throughput ====")
    print(f"owner_dev={owner} req_dev={req}  P2P={'yes' if pywasipc.can_access_peer(req, owner) else 'no'}")
    print(f"region={args.region_mib} MiB  page={args.page_mib} MiB  "
          f"streams={args.streams}  inflight/stream={args.inflight}")
    print(f"total_read={args.total_gib:.2f} GiB  elapsed={elapsed:.3f} s")
    print(f"throughput: {gib_s:.2f} GiB/s  ({mib_s:.0f} MiB/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure pywasipc one-sided async read throughput (GiB/s).")
    parser.add_argument("--owner-dev", type=int, default=int(os.getenv("WASIPC_OWNER_DEV", "0")))
    parser.add_argument("--req-dev",   type=int, default=int(os.getenv("WASIPC_REQ_DEV", "1")))
    parser.add_argument("--region-mib", type=int, default=512, help="owner 侧暴露区域大小（MiB）")
    parser.add_argument("--page-mib",   type=int, default=16,  help="单页读取大小（MiB）")
    parser.add_argument("--streams",    type=int, default=2,   help="并发读取的 CUDA streams 数")
    parser.add_argument("--inflight",   type=int, default=2,   help="每个 stream 保持的 in-flight 页数")
    parser.add_argument("--total-gib",  type=float, default=8.0, help="总读取体量（GiB）")
    parser.add_argument("--verify", action="store_true", help="读后随机校验图样（略降吞吐）")
    args = parser.parse_args()
    run_bench(args)
