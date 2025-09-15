import pytest
import torch
import pywasipc

def test_bad_handle_bytes_raises(devices):
    _, req = devices
    torch.cuda.set_device(req)
    with pytest.raises(RuntimeError):
        pywasipc.open_region(b"\x00\x01\x02")  # 错误长度

def test_out_of_range_read_raises(devices, sizes, owner_process, requester_info):
    _, req = devices
    REGION_BYTES, _ = sizes
    torch.cuda.set_device(req)
    dst = torch.empty(4, dtype=torch.uint8, device=f"cuda:{req}")
    # 故意越界：src_off 放尾部，bytes 溢出
    with pytest.raises(RuntimeError):
        pywasipc.read_async(requester_info, dst.data_ptr(), 8, REGION_BYTES-2, 0)
