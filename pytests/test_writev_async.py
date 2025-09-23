# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import pywasipc

def test_write_async_roundtrip(devices, sizes, owner_process, requester_info, expect_slice):
    """
    requester：将 pattern 写入 owner 的远端 region，然后从远端再读回校验。
    覆盖：pywasipc.write_async_tensor + read_async
    """
    owner, req = devices
    REGION_BYTES, PAGE_BYTES = sizes
    info = requester_info  # RegionInfo (远端 owner 的内存)
    torch.cuda.set_device(req)

    # 写入的 payload：从 0 偏移开始，长度为 REGION_BYTES（与 owner 端初始化的花纹不同）
    payload = expect_slice(REGION_BYTES, src_off=0, device=f"cuda:{req}")

    # 写：把 requester 的 payload 写到 owner region
    stream = torch.cuda.current_stream().cuda_stream
    pywasipc.write_async_tensor(info, payload, REGION_BYTES, 0, stream)
    torch.cuda.current_stream().synchronize()

    # 读回：从 owner region 读到 requester 本地 buffer
    out = torch.empty_like(payload)
    pywasipc.read_async(info, out.data_ptr(), REGION_BYTES, 0, stream)
    torch.cuda.current_stream().synchronize()

    assert torch.equal(out, payload), "roundtrip mismatch after write_async"


def test_writev_async_two_segments(devices, sizes, owner_process, requester_info, expect_slice):
    """
    requester：分两段写入 owner 的远端 region 的不同 offset，然后整体读回校验。
    覆盖：pywasipc.writev_async_tensor
    """
    owner, req = devices
    REGION_BYTES, PAGE_BYTES = sizes
    info = requester_info
    torch.cuda.set_device(req)

    # 切两段：前半 seg0，后半 seg1，对应连续的 offset
    half = REGION_BYTES // 2
    seg0 = expect_slice(half, src_off=0, device=f"cuda:{req}")
    seg1 = expect_slice(REGION_BYTES - half, src_off=half, device=f"cuda:{req}")

    stream = torch.cuda.current_stream().cuda_stream
    pywasipc.writev_async_tensor(
        info,
        [
            (seg0, seg0.numel() * seg0.element_size(), 0),
            (seg1, seg1.numel() * seg1.element_size(), half),
        ],
        stream,
    )
    torch.cuda.current_stream().synchronize()

    # 读回并校验
    out = torch.empty(REGION_BYTES, dtype=torch.uint8, device=f"cuda:{req}")
    pywasipc.read_async(info, out.data_ptr(), REGION_BYTES, 0, stream)
    torch.cuda.current_stream().synchronize()

    expect = torch.cat([seg0, seg1], dim=0)
    assert torch.equal(out, expect), "mismatch after writev_async"
