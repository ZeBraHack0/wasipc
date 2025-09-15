# pytests/test_readv_async.py

import torch
import pywasipc

def test_readv_async_multi_pages(devices, sizes, owner_process, requester_info, expect_slice):
    owner, req = devices
    REGION_BYTES, PAGE_BYTES = sizes

    torch.cuda.set_device(req)
    s0 = torch.cuda.Stream(device=req)
    s1 = torch.cuda.Stream(device=req)

    p0 = torch.empty(PAGE_BYTES, dtype=torch.uint8, device=f"cuda:{req}")
    p1 = torch.empty(PAGE_BYTES, dtype=torch.uint8, device=f"cuda:{req}")
    descs = [
        (p0.data_ptr(), PAGE_BYTES, 0,        s0.cuda_stream),
        (p1.data_ptr(), PAGE_BYTES, 16<<20,   s1.cuda_stream),
    ]
    pywasipc.readv_async(requester_info, descs)
    s0.synchronize(); s1.synchronize()

    ref0 = expect_slice(PAGE_BYTES, 0, device=f"cuda:{req}")
    ref1 = expect_slice(PAGE_BYTES, 16<<20, device=f"cuda:{req}")
    assert torch.equal(p0, ref0)
    assert torch.equal(p1, ref1)

def test_stream_semantics(devices, sizes, owner_process, requester_info):
    _, req = devices
    _, PAGE_BYTES = sizes
    torch.cuda.set_device(req)
    non_default = torch.cuda.Stream(device=req)
    dst = torch.empty(PAGE_BYTES, dtype=torch.uint8, device=f"cuda:{req}")
    evt = torch.cuda.Event(blocking=False, interprocess=False)

    with torch.cuda.stream(non_default):
        pywasipc.read_async(requester_info, dst.data_ptr(), PAGE_BYTES, 0, non_default.cuda_stream)
        evt.record(non_default)

    evt.synchronize()
    assert int(dst.sum().item()) != 0
