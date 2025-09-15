# pywasipc

CUDA IPC + P2P one-sided reads for WAS.

## Install
```shell
python -m pip install -v .
```

## Test
```shell
python -m pytest -q pytests
```

## Quickstart
```python
import torch, pywasipc

# ---------- Owner 侧 ----------
torch.cuda.set_device(0)
# 举例：owner 已有一个装满权重的 device tensor flat_buf
flat_buf = torch.empty((256<<20)//2, dtype=torch.bfloat16, device="cuda")  # 256MiB 演示
handle_bytes = pywasipc.register_region(flat_buf.data_ptr(), flat_buf.numel()*2)  # bytes

# 把 handle_bytes 通过 torch.distributed.broadcast / 文件 / RPC 发给 requester...
# 这里只是示意：
open("/tmp/ffn.handle","wb").write(handle_bytes)

# ---------- Requester 侧 ----------
torch.cuda.set_device(1)
handle_bytes = open("/tmp/ffn.handle","rb").read()
# 先检查/启用 peer：
assert pywasipc.can_access_peer(1, 0), "P2P not available between dev1 and dev0"
pywasipc.enable_peer_access(1, 0)

info = pywasipc.open_region(handle_bytes)     # RegionInfo
page = torch.empty((16<<20)//2, dtype=torch.bfloat16, device="cuda")  # 16MiB
stream = torch.cuda.current_stream()
pywasipc.read_async(info, page.data_ptr(), page.numel()*2, src_off=0,
                    stream_ptr=stream.cuda_stream)

# …继续在 stream 上排更多页；与 compute stream 做双缓冲
torch.cuda.synchronize()

pywasipc.close_region(info)

# ---------- Owner 清理 ----------
# 全部 requester close 后：
pywasipc.deregister_region(flat_buf.data_ptr())
```
