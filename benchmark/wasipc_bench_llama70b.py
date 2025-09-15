#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wasipc_bench_llama70b.py — 模拟 Llama3.1-70B (80 层, DP=2) 的 WAS 读取形态：
每个 owner 暴露 40 个 FFN region（默认每个 512 MiB），requester 侧把这 40 个 region
全部按“分页 + 多流 + in-flight”完整读取一遍（或多遍），测量聚合吞吐。

示例（贴近实际）：
  python tools/wasipc_bench_llama70b.py --owner-dev 0 --req-dev 1 \
    --regions 40 --region-mib 512 --page-mib 16 \
    --streams 3 --inflight 3 --passes 1

如需更严一点：
  --region-mib 672  （如果你按我们之前估算的 Llama70B 单卡 TP=2 的 FFN ≈ 672 MiB）
"""

import argparse
import os
import time
import itertools
import multiprocessing as mp
from typing import List

import torch
import pywasipc

mp.set_start_method("spawn", force=True)

def mib(x): return x * (1 << 20)
def gib(x): return x * (1 << 30)

def make_pattern(count_bytes: int, offset: int = 0, device: str = "cpu"):
    # 简单 uint8 张量：(i+offset)*7 mod 256
    i0 = offset
    t = torch.arange(i0, i0 + count_bytes, dtype=torch.int64, device=device)
    return (t.to(torch.uint8) * 7).to(torch.uint8)

def owner_proc(dev: int, regions: int, region_bytes: int, q_handles: mp.Queue,
               ev_stop: mp.Event, verify: bool):
    """
    owner 子进程：在指定 GPU 上分配 N 个连续显存缓冲作为 N 个 region，
    逐个 register，并把句柄列表通过队列发回。结束时等待 ev_stop 再清理。
    """
    torch.cuda.set_device(dev)
    bufs = []
    handles: List[bytes] = []
    try:
        for r in range(regions):
            buf = torch.empty(region_bytes, dtype=torch.uint8, device=f"cuda:{dev}")
            if verify:
                # 每个 region 用不同的 base offset，避免张量重复
                base_off = r * region_bytes
                buf.copy_(make_pattern(region_bytes, base_off, device=f"cuda:{dev}"))
            handle = pywasipc.register_region(buf.data_ptr(), region_bytes)
            bufs.append(buf)
            handles.append(handle)
        q_handles.put(handles)
        ev_stop.wait()
    finally:
        # 注销并释放
        for buf in bufs:
            try:
                pywasipc.deregister_region(buf.data_ptr())
            except Exception:
                pass
        for buf in bufs:
            del buf
        torch.cuda.synchronize(dev)

def run_bench(args):
    owner, req = args.owner_dev, args.req_dev
    regions = args.regions
    region_bytes = mib(args.region_mib)
    page_bytes   = mib(args.page_mib)
    assert region_bytes % page_bytes == 0, "region-mib 必须能被 page-mib 整除"

    # 总读取字节 = regions × region_bytes × passes
    total_bytes  = regions * region_bytes * max(1, args.passes)

    # 启动 owner 子进程
    q = mp.Queue(maxsize=1)
    ev = mp.Event()
    p = mp.Process(target=owner_proc,
                   args=(owner, regions, region_bytes, q, ev, args.verify),
                   daemon=True)
    p.start()
    handles: List[bytes] = q.get(timeout=30.0)

    # requester 准备
    torch.cuda.set_device(req)
    if not pywasipc.can_access_peer(req, owner):
        raise RuntimeError(f"P2P not available between dev{req} and dev{owner}")
    pywasipc.enable_peer_access(req, owner)

    infos = [pywasipc.open_region(h) for h in handles]  # N 个 RegionInfo
    # 分配流与 in-flight 页缓冲
    streams = [torch.cuda.Stream(device=req) for _ in range(args.streams)]
    pages   = [[torch.empty(page_bytes, dtype=torch.uint8, device=f"cuda:{req}")
                for _ in range(args.inflight)] for __ in range(args.streams)]

    # 预热：对前几个 region 读一页
    warm_regions = min(regions, len(streams))
    for s_idx, s in enumerate(streams[:warm_regions]):
        with torch.cuda.stream(s):
            pywasipc.read_async(infos[s_idx], pages[s_idx][0].data_ptr(),
                                page_bytes, 0, s.cuda_stream)
    for s in streams[:warm_regions]: s.synchronize()

    # ===== 正式调度：多 region × 分页 × 多流 × in-flight =====
    # round-robin 依次覆盖 region_idx = 0..regions-1，再从头开始，做 passes 遍
    region_order = list(range(regions))
    round_robin  = itertools.chain.from_iterable([region_order] * max(1, args.passes))

    # 为每个 region 维护“下一页偏移”
    next_off = [0] * regions
    pages_per_region = region_bytes // page_bytes

    issued_pages = 0
    target_pages = (total_bytes // page_bytes)

    # 为了 pipeline，外层轮流给每个 stream 塞任务；每个 stream 内保持 inflight 个未完成页
    start = time.perf_counter()
    rr_iter = iter(round_robin)
    # 当前要处理的 region（每个 stream 一个“当前 region”游标）
    cur_region = [next(rr_iter) for _ in range(args.streams)]

    while issued_pages < target_pages:
        for s_idx, s in enumerate(streams):
            # 这一小段给当前 stream 塞满 inflight
            for k in range(args.inflight):
                if issued_pages >= target_pages:
                    break
                ridx = cur_region[s_idx]
                off  = next_off[ridx]
                # 发起一次页读
                with torch.cuda.stream(s):
                    pywasipc.read_async(infos[ridx], pages[s_idx][k].data_ptr(),
                                        page_bytes, off, s.cuda_stream)
                issued_pages += 1
                # 更新 region 的 next_off；若整层已读完，切到下一个 region
                off += page_bytes
                if off >= region_bytes:
                    next_off[ridx] = 0
                    try:
                        cur_region[s_idx] = next(rr_iter)
                    except StopIteration:
                        # 所有 passes 发完
                        break
                else:
                    next_off[ridx] = off
        # 继续下一轮，把所有 streams 继续填充

    # 等所有流结束
    for s in streams: s.synchronize()
    elapsed = time.perf_counter() - start

    # 校验（可选）：对少数 region/页 spot check
    if args.verify:
        # 从前两个 region 抽样校验头/尾页（不影响统计，因为统计已结束）
        torch.cuda.synchronize(req)
        check_regions = min(2, regions)
        for ridx in range(check_regions):
            info = infos[ridx]
            # 头页
            pywasipc.read_async(info, pages[0][0].data_ptr(), page_bytes, 0,
                                torch.cuda.current_stream().cuda_stream)
            # 尾页
            pywasipc.read_async(info, pages[0][1].data_ptr(), page_bytes, region_bytes - page_bytes,
                                torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize(req)
            base = ridx * region_bytes
            ref0 = make_pattern(page_bytes, base + 0, device=f"cuda:{req}")
            ref1 = make_pattern(page_bytes, base + (region_bytes - page_bytes), device=f"cuda:{req}")
            assert torch.equal(pages[0][0], ref0), f"pattern mismatch at region {ridx} head"
            assert torch.equal(pages[0][1], ref1), f"pattern mismatch at region {ridx} tail"

    # 清理
    for info in infos:
        try: pywasipc.close_region(info)
        except Exception: pass
    ev.set()
    p.join(timeout=10.0)

    # 统计
    gib_s = total_bytes / (1<<30) / elapsed
    mib_s = total_bytes / (1<<20) / elapsed
    print("==== WASIPC Multi-Region Async Read Throughput ====")
    print(f"owner_dev={owner}  req_dev={req}  P2P={'yes' if pywasipc.can_access_peer(req, owner) else 'no'}")
    print(f"regions={regions}  region={args.region_mib} MiB  page={args.page_mib} MiB")
    print(f"streams={args.streams}  inflight/stream={args.inflight}  passes={args.passes}")
    print(f"total_read={total_bytes/(1<<30):.2f} GiB  elapsed={elapsed:.3f} s")
    print(f"throughput: {gib_s:.2f} GiB/s  ({mib_s:.0f} MiB/s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-region WAS IPC throughput benchmark (Llama-70B, DP=2 style).")
    parser.add_argument("--owner-dev", type=int, default=int(os.getenv("WASIPC_OWNER_DEV", "0")))
    parser.add_argument("--req-dev",   type=int, default=int(os.getenv("WASIPC_REQ_DEV", "1")))
    parser.add_argument("--regions",   type=int, default=40, help="该 owner 暴露的 region 数（Llama70B, DP=2 → 40）")
    parser.add_argument("--region-mib",type=int, default=512, help="每个 region 大小（MiB），可改成 672 以符合我们之前估算")
    parser.add_argument("--page-mib",  type=int, default=16,  help="单页读取大小（MiB）")
    parser.add_argument("--streams",   type=int, default=3,   help="并发读取的 CUDA streams 数")
    parser.add_argument("--inflight",  type=int, default=3,   help="每个 stream 保持的 in-flight 页数")
    parser.add_argument("--passes",    type=int, default=1,   help="把所有 regions 全部读完的遍数")
    parser.add_argument("--verify", action="store_true", help="随机校验（开了会略降吞吐）")
    args = parser.parse_args()
    run_bench(args)
